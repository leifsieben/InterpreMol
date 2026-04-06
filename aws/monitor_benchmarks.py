#!/usr/bin/env python3
"""
Monitor MoleculeNet benchmark runs on the EC2 node.

Reports:
- whether the benchmark process is alive
- which phase/run is being tracked
- completed jobs vs expected jobs
- completed dataset/split/model groups
- coarse ETA based on finished-job throughput
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_HOST = "ubuntu@3.220.174.83"
DEFAULT_KEY = "~/.ssh/interpremol-key.pem"
DEFAULT_BENCH_ROOT = "/home/ubuntu/interpremol_benchmarks"


REMOTE_PY = r"""
import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

bench_root = Path(os.environ["BENCH_ROOT"])
phase = os.environ.get("PHASE", "phase1")
explicit_run = os.environ.get("RUN_DIR")

if explicit_run:
    run_dir = Path(explicit_run)
else:
    candidates = sorted(bench_root.glob("moleculenet_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    run_dir = candidates[-1] if candidates else bench_root / "missing"

log_candidates = sorted((bench_root / "logs").glob(f"{phase}_*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
log_path = log_candidates[-1] if log_candidates else None

phase_cfg = {
    "phase1": {
        "datasets": ["bbbp", "bace", "clintox", "hiv", "muv", "pcba", "sider", "tox21", "toxcast", "esol", "freesolv", "lipo", "qm7", "qm8", "qm9"],
        "splits": ["random", "scaffold"],
        "seeds": [0, 1, 2],
        "models": ["interpremol_frozen", "chemeleon_frozen", "chemeleon_finetune", "random_forest"],
    },
    "phase2": {
        "datasets": ["bbbp", "bace", "clintox", "hiv", "muv", "pcba", "sider", "tox21", "toxcast", "esol", "freesolv", "lipo", "qm7", "qm8", "qm9"],
        "splits": ["random", "scaffold"],
        "seeds": [0, 1, 2],
        "models": ["interpremol_finetune"],
    },
}
cfg = phase_cfg[phase]
expected_jobs = len(cfg["datasets"]) * len(cfg["splits"]) * len(cfg["seeds"]) * len(cfg["models"])

summary = {
    "phase": phase,
    "run_dir": str(run_dir),
    "run_name": run_dir.name,
    "log_path": str(log_path) if log_path else None,
    "expected_jobs": expected_jobs,
}

try:
    pgrep_out = subprocess.check_output(
        ["pgrep", "-af", f"python -m benchmarks.run_moleculenet"],
        text=True,
    ).strip()
    lines = [line for line in pgrep_out.splitlines() if line.strip()]
except subprocess.CalledProcessError:
    lines = []

phase_processes = [line for line in lines if phase in line or "benchmarks.run_moleculenet" in line]
summary["live_process_count"] = len(phase_processes)
summary["working"] = len(phase_processes) > 0

summary_csv = run_dir / "summary.csv"
rows = []
if summary_csv.exists():
    with open(summary_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
summary["completed_jobs"] = len(rows)
summary["summary_exists"] = summary_csv.exists()

completed_groups = {}
for row in rows:
    key = (row["dataset"], row["split"], row["model"])
    completed_groups.setdefault(key, set()).add(row["seed"])
summary["completed_groups"] = sum(1 for seeds in completed_groups.values() if len(seeds) == 3)

dataset_progress = {}
for row in rows:
    dataset_progress.setdefault(row["dataset"], 0)
    dataset_progress[row["dataset"]] += 1
summary["dataset_progress"] = dict(sorted(dataset_progress.items()))

tail_text = ""
if log_path and log_path.exists():
    try:
        tail_text = subprocess.check_output(["tail", "-n", "60", str(log_path)], text=True)
    except Exception:
        tail_text = ""
summary["tail_excerpt"] = tail_text[-4000:]

elapsed_sec = None
if log_path and log_path.exists():
    elapsed_sec = max(0.0, datetime.now(timezone.utc).timestamp() - log_path.stat().st_mtime)
    # Better approximation: if process is alive, use log file creation-ish time if available.
    try:
        st = log_path.stat()
        elapsed_sec = max(0.0, datetime.now(timezone.utc).timestamp() - st.st_ctime)
    except Exception:
        pass
summary["elapsed_sec"] = elapsed_sec

if rows and expected_jobs > len(rows) and elapsed_sec and elapsed_sec > 0:
    rate = len(rows) / elapsed_sec
    summary["eta_sec"] = (expected_jobs - len(rows)) / rate if rate > 0 else None
else:
    summary["eta_sec"] = None

summary["status"] = "completed" if summary["completed_jobs"] >= expected_jobs else ("running" if summary["working"] else "idle")

try:
    gpu_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if gpu_csv:
        util, mem_used, mem_total = [part.strip() for part in gpu_csv[0].split(",")]
        summary["gpu"] = {
            "gpu_util_pct": int(util),
            "gpu_mem_used_mib": int(mem_used),
            "gpu_mem_total_mib": int(mem_total),
        }
except Exception:
    summary["gpu"] = None

print(json.dumps(summary))
"""


def run_remote_summary(host: str, key: str, bench_root: str, phase: str, run_dir: str | None) -> dict[str, Any]:
    key = str(Path(key).expanduser())
    cmd = [
        "ssh",
        "-i",
        key,
        host,
        (
            f"BENCH_ROOT={json.dumps(bench_root)} "
            + f"PHASE={json.dumps(phase)} "
            + (f"RUN_DIR={json.dumps(run_dir)} " if run_dir else "")
            + "python3 - <<'PY'\n"
            + REMOTE_PY
            + "\nPY"
        ),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def format_status(summary: dict[str, Any]) -> str:
    lines = [
        f"Phase: {summary.get('phase', 'unknown')}",
        f"Run: {summary.get('run_name', 'unknown')}",
        f"Run dir: {summary.get('run_dir', 'unknown')}",
        f"Log: {summary.get('log_path', 'unknown')}",
        f"Status: {summary.get('status', 'unknown')}",
        f"Working: {'yes' if summary.get('working') else 'no'}",
        f"Progress: {summary.get('completed_jobs', 0)}/{summary.get('expected_jobs', 0)} jobs",
        f"Completed groups (3 seeds): {summary.get('completed_groups', 0)}",
        f"Elapsed: {format_duration(summary.get('elapsed_sec'))}",
        f"ETA: {format_duration(summary.get('eta_sec'))}",
    ]

    gpu = summary.get("gpu") or {}
    if gpu:
        lines.append(
            "GPU: "
            f"{gpu.get('gpu_util_pct', '?')}% util, "
            f"{gpu.get('gpu_mem_used_mib', '?')}/{gpu.get('gpu_mem_total_mib', '?')} MiB"
        )

    dataset_progress = summary.get("dataset_progress") or {}
    if dataset_progress:
        top = ", ".join(f"{k}={v}" for k, v in list(dataset_progress.items())[:8])
        lines.append(f"Dataset progress: {top}")

    tail = summary.get("tail_excerpt", "").strip()
    if tail:
        lines.append("")
        lines.append("Recent log tail:")
        lines.append(tail)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor MoleculeNet benchmark progress on the EC2 node.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--run-dir", default=None, help="Optional explicit remote run directory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_remote_summary(
        host=args.host,
        key=args.key,
        bench_root=args.bench_root,
        phase=args.phase,
        run_dir=args.run_dir,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_status(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
