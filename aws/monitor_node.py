#!/usr/bin/env python3
"""
Monitor an InterpreMol run on the EC2 node.

Reports:
- whether the node/run appears alive
- current phase (HPO vs full training)
- progress (e.g. x/16 trials)
- coarse ETA when a best-effort estimate is possible
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HOST = "ubuntu@3.220.174.83"
DEFAULT_KEY = "~/.ssh/interpremol-key.pem"
DEFAULT_RUNS_ROOT = "/home/ubuntu/interpremol_runs"


REMOTE_PY = r"""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

runs_root = Path(os.environ["RUNS_ROOT"])
run_dir_arg = os.environ.get("RUN_DIR")
run_dir = Path(run_dir_arg) if run_dir_arg else runs_root / "latest"
run_dir = run_dir.resolve()

summary = {
    "run_dir": str(run_dir),
    "exists": run_dir.exists(),
}

if not run_dir.exists():
    print(json.dumps(summary))
    raise SystemExit(0)

summary["run_name"] = run_dir.name

heartbeat_path = run_dir / "heartbeats" / "heartbeat.json"
if heartbeat_path.exists():
    heartbeat = json.load(open(heartbeat_path))
    summary["heartbeat"] = heartbeat
    try:
        hb_time = datetime.fromisoformat(heartbeat["timestamp"].replace("Z", "+00:00"))
        summary["heartbeat_age_sec"] = (datetime.now(timezone.utc) - hb_time).total_seconds()
    except Exception:
        summary["heartbeat_age_sec"] = None
else:
    summary["heartbeat"] = None
    summary["heartbeat_age_sec"] = None

config_path = run_dir / "config.json"
summary["config"] = json.load(open(config_path)) if config_path.exists() else {}

train_log = run_dir / "logs" / "train.out"
tail_text = ""
if train_log.exists():
    try:
        tail_text = subprocess.check_output(
            ["tail", "-n", "200", str(train_log)],
            text=True,
        )
    except Exception:
        tail_text = ""

summary["tail_excerpt"] = tail_text[-4000:]

hpo_dir = run_dir / "checkpoints" / "interpremol_hyperopt"
if hpo_dir.exists():
    summary["phase"] = "hpo"
    state_files = sorted(hpo_dir.glob("experiment_state-*.json"), key=lambda p: p.stat().st_mtime)
    trial_status_counts = {}
    trials = []
    start_time = None
    if state_files:
        obj = json.load(open(state_files[-1]))
        for trial_blob, runtime_blob in obj.get("trial_data", []):
            trial = json.loads(trial_blob)
            runtime = json.loads(runtime_blob)
            status = trial.get("status", "UNKNOWN")
            trial_status_counts[status] = trial_status_counts.get(status, 0) + 1
            trial_info = {
                "trial_id": trial.get("trial_id"),
                "status": status,
                "start_time": runtime.get("start_time"),
                "num_failures": runtime.get("num_failures", 0),
            }
            trials.append(trial_info)
            st = runtime.get("start_time")
            if isinstance(st, (int, float)):
                start_time = st if start_time is None else min(start_time, st)
    total = len(trials)
    finished = sum(1 for t in trials if t["status"] in {"TERMINATED", "ERROR"})
    running = sum(1 for t in trials if t["status"] == "RUNNING")
    pending = total - finished - running
    summary["hpo"] = {
        "total_trials": total,
        "finished_trials": finished,
        "running_trials": running,
        "pending_trials": pending,
        "status_counts": trial_status_counts,
    }
    summary["status"] = "failed" if total and trial_status_counts.get("ERROR", 0) == total else (
        "completed" if total and finished == total and trial_status_counts.get("ERROR", 0) == 0 else (
            "running" if running or pending else "unknown"
        )
    )
    if start_time is not None:
        elapsed = max(0.0, datetime.now(timezone.utc).timestamp() - start_time)
        summary["elapsed_sec"] = elapsed
        remaining = max(total - finished, 0)
        summary["eta_sec"] = (elapsed / finished * remaining) if finished > 0 and remaining > 0 and summary["status"] == "running" else None
    else:
        summary["elapsed_sec"] = None
        summary["eta_sec"] = None
else:
    summary["phase"] = "full_training"
    epochs_total = summary["config"].get("epochs")
    ckpts = []
    for p in (run_dir / "checkpoints").glob("checkpoint_epoch_*.pt"):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt", p.name)
        if m:
            ckpts.append(int(m.group(1)))
    current_epoch = max(ckpts) if ckpts else 0
    m = re.findall(r"Epoch (\d+)/(\d+)", tail_text)
    if m:
        last_epoch, total_from_log = m[-1]
        current_epoch = max(current_epoch, int(last_epoch))
        if not epochs_total:
            epochs_total = int(total_from_log)
    summary["training"] = {
        "current_epoch": current_epoch,
        "total_epochs": epochs_total,
    }
    results_path = run_dir / "checkpoints" / "results.json"
    if results_path.exists():
        results = json.load(open(results_path))
        current_epoch = max(current_epoch, int(results.get("final_epoch", current_epoch)))
        summary["training"]["current_epoch"] = current_epoch
        summary["status"] = "completed"
    else:
        summary["status"] = "running" if current_epoch > 0 else "starting"
    if summary["heartbeat"] and summary["heartbeat_age_sec"] is not None:
        start_guess = summary["heartbeat_age_sec"]
    else:
        start_guess = None
    if current_epoch > 0 and epochs_total and start_guess is not None:
        summary["eta_sec"] = (start_guess / current_epoch) * max(epochs_total - current_epoch, 0)
    else:
        summary["eta_sec"] = None

print(json.dumps(summary))
"""


def run_remote_summary(host: str, key: str, runs_root: str, run_dir: str | None) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-i",
        key,
        host,
        (
            f"RUNS_ROOT={json.dumps(runs_root)} "
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
    hb_age = summary.get("heartbeat_age_sec")
    heartbeat = summary.get("heartbeat") or {}
    alive = hb_age is not None and hb_age < 180
    working = heartbeat.get("python_pretrain_processes", 0) > 0
    phase = summary.get("phase", "unknown")
    status = summary.get("status", "unknown")

    lines = [
        f"Run: {summary.get('run_name', 'unknown')}",
        f"Path: {summary.get('run_dir', 'unknown')}",
        f"Node alive: {'yes' if alive else 'no'}",
        f"Working: {'yes' if working else 'no'}",
        f"Heartbeat age: {format_duration(hb_age)}",
        f"Phase: {phase}",
        f"Status: {status}",
    ]

    if heartbeat:
        lines.append(
            "GPU: "
            f"{heartbeat.get('gpu_util_pct', '?')}% util, "
            f"{heartbeat.get('gpu_mem_used_mib', '?')}/{heartbeat.get('gpu_mem_total_mib', '?')} MiB"
        )

    if phase == "hpo":
        hpo = summary.get("hpo", {})
        total = hpo.get("total_trials", 0)
        finished = hpo.get("finished_trials", 0)
        running = hpo.get("running_trials", 0)
        pending = hpo.get("pending_trials", 0)
        counts = hpo.get("status_counts", {})
        lines.append(f"Progress: {finished}/{total} trials finished")
        lines.append(
            "Trials: "
            f"running={running}, pending={pending}, "
            f"terminated={counts.get('TERMINATED', 0)}, error={counts.get('ERROR', 0)}"
        )
    elif phase == "full_training":
        training = summary.get("training", {})
        lines.append(
            f"Progress: epoch {training.get('current_epoch', 0)}/{training.get('total_epochs', '?')}"
        )

    lines.append(f"ETA: {format_duration(summary.get('eta_sec'))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor InterpreMol EC2 node progress")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH host")
    parser.add_argument("--key", default=DEFAULT_KEY, help="SSH private key path")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT, help="Remote runs root")
    parser.add_argument("--run-dir", help="Specific remote run dir; defaults to latest symlink")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    key = str(Path(args.key).expanduser())
    try:
        summary = run_remote_summary(args.host, key, args.runs_root, args.run_dir)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or str(exc))
        return exc.returncode or 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_status(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
