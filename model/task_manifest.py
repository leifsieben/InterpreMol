"""
Task audit and manifest utilities for multi-task pretraining datasets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - pyarrow is expected in the training env
    pq = None


SMILES_COLUMNS = {"SMILES", "SMILES_std"}


def split_task_column(task_name: str) -> Dict[str, str]:
    """Infer broad family and subfamily from a task column name."""
    family = task_name.split("__", 1)[0]

    if family.startswith("L1000_"):
        return {"broad_family": "L1000", "subfamily": family}

    return {"broad_family": family, "subfamily": family}


def infer_task_type(unique_values: List[float]) -> Dict[str, Any]:
    """Infer task type from the observed unique values."""
    value_set = set(unique_values)

    if value_set.issubset({0.0, 1.0}):
        return {"task_type": "binary", "num_classes": 2}
    if value_set.issubset({0.0, 1.0, 2.0}):
        return {"task_type": "multiclass", "num_classes": 3}

    return {"task_type": "unknown", "num_classes": len(unique_values)}


def default_include_flags(task_type: str) -> Dict[str, bool]:
    """Default inclusion policy for new manifests."""
    include = task_type in {"binary", "multiclass"}
    return {
        "include_in_hpo": include,
        "include_in_stage2": include,
    }


def get_label_columns(data_file: str, smiles_col: str) -> List[str]:
    """Return all non-SMILES label columns from a file."""
    path = Path(data_file)
    if path.suffix == ".parquet" and pq is not None:
        schema = pq.ParquetFile(path).schema_arrow
        return [field.name for field in schema if field.name not in SMILES_COLUMNS and field.name != smiles_col]

    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=0)
        return [c for c in df.columns if c not in SMILES_COLUMNS and c != smiles_col]

    raise ValueError(f"Unsupported dataset format for task audit: {data_file}")


def _read_column(data_file: str, column_name: str) -> pd.Series:
    path = Path(data_file)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=[column_name])[column_name]
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=[column_name])[column_name]
    raise ValueError(f"Unsupported dataset format for task audit: {data_file}")


def _init_task_stat(task_name: str) -> Dict[str, Any]:
    return {
        "task_name": task_name,
        "valid_count": 0,
        "min_value": None,
        "max_value": None,
        "class_counts": Counter(),
    }


def _update_task_stat(stat: Dict[str, Any], values: np.ndarray) -> None:
    if values.size == 0:
        return

    stat["valid_count"] += int(values.size)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    stat["min_value"] = value_min if stat["min_value"] is None else min(stat["min_value"], value_min)
    stat["max_value"] = value_max if stat["max_value"] is None else max(stat["max_value"], value_max)
    stat["class_counts"].update(values.tolist())


def _finalize_task_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    unique_values = sorted(stat["class_counts"].keys())
    family_info = split_task_column(stat["task_name"])
    type_info = infer_task_type(unique_values)
    entry = {
        "task_name": stat["task_name"],
        "valid_count": stat["valid_count"],
        "min_value": stat["min_value"],
        "max_value": stat["max_value"],
        "unique_values": unique_values,
        "class_counts": {
            str(int(k) if float(k).is_integer() else k): int(v)
            for k, v in sorted(stat["class_counts"].items())
        },
        **family_info,
        **type_info,
    }
    entry.update(default_include_flags(entry["task_type"]))
    return entry


def _build_parquet_task_manifest(data_file: str, smiles_col: str, label_cols: List[str], task_chunk_size: int) -> Dict[str, Any]:
    parquet_file = pq.ParquetFile(data_file)
    stats = {task_name: _init_task_stat(task_name) for task_name in label_cols}

    for rg_idx in range(parquet_file.metadata.num_row_groups):
        for chunk_start in range(0, len(label_cols), task_chunk_size):
            chunk_cols = label_cols[chunk_start:chunk_start + task_chunk_size]
            table = parquet_file.read_row_group(rg_idx, columns=chunk_cols)

            for col_name in chunk_cols:
                col_np = table.column(col_name).to_numpy(zero_copy_only=False)
                valid_mask = ~np.isnan(col_np)
                _update_task_stat(stats[col_name], col_np[valid_mask].astype(np.float64, copy=False))

    tasks = [_finalize_task_stat(stats[task_name]) for task_name in label_cols]
    return {
        "data_file": str(Path(data_file).resolve()),
        "smiles_col": smiles_col,
        "n_tasks": len(tasks),
        "tasks": tasks,
    }


def audit_task_column(data_file: str, task_name: str) -> Dict[str, Any]:
    """Audit one task column and return its manifest entry."""
    series = _read_column(data_file, task_name).dropna()
    values = series.astype(float)
    unique_values = sorted(values.unique().tolist())
    counts = Counter(values.tolist())
    family_info = split_task_column(task_name)
    type_info = infer_task_type(unique_values)

    entry = {
        "task_name": task_name,
        "valid_count": int(values.shape[0]),
        "min_value": float(values.min()) if not values.empty else None,
        "max_value": float(values.max()) if not values.empty else None,
        "unique_values": unique_values,
        "class_counts": {str(int(k) if float(k).is_integer() else k): int(v) for k, v in sorted(counts.items())},
        **family_info,
        **type_info,
    }
    entry.update(default_include_flags(entry["task_type"]))
    return entry


def build_task_manifest(data_file: str, smiles_col: str = "SMILES_std", task_chunk_size: int = 128) -> Dict[str, Any]:
    """Build a task manifest for a wide supervised dataset."""
    label_cols = get_label_columns(data_file, smiles_col)
    path = Path(data_file)

    if path.suffix == ".parquet" and pq is not None:
        manifest = _build_parquet_task_manifest(data_file, smiles_col, label_cols, task_chunk_size)
        tasks = manifest["tasks"]
    else:
        tasks = [audit_task_column(data_file, task_name) for task_name in label_cols]
        manifest = {
            "data_file": str(Path(data_file).resolve()),
            "smiles_col": smiles_col,
            "n_tasks": len(tasks),
            "tasks": tasks,
        }

    family_summary: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        family = task["broad_family"]
        family_summary.setdefault(
            family,
            {
                "n_tasks": 0,
                "task_types": Counter(),
                "include_in_hpo": 0,
                "include_in_stage2": 0,
            },
        )
        family_summary[family]["n_tasks"] += 1
        family_summary[family]["task_types"][task["task_type"]] += 1
        family_summary[family]["include_in_hpo"] += int(task["include_in_hpo"])
        family_summary[family]["include_in_stage2"] += int(task["include_in_stage2"])

    manifest["family_summary"] = _normalize_family_summary(family_summary)
    return manifest


def _normalize_family_summary(family_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        family: {
            **summary,
            "task_types": dict(summary["task_types"]),
        }
        for family, summary in family_summary.items()
    }


def recompute_family_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    family_summary: Dict[str, Dict[str, Any]] = {}
    tasks = manifest["tasks"]
    for task in tasks:
        family = task["broad_family"]
        family_summary.setdefault(
            family,
            {
                "n_tasks": 0,
                "task_types": Counter(),
                "include_in_hpo": 0,
                "include_in_stage2": 0,
            },
        )
        family_summary[family]["n_tasks"] += 1
        family_summary[family]["task_types"][task["task_type"]] += 1
        family_summary[family]["include_in_hpo"] += int(task["include_in_hpo"])
        family_summary[family]["include_in_stage2"] += int(task["include_in_stage2"])
    manifest["family_summary"] = _normalize_family_summary(family_summary)
    return manifest


def save_task_manifest(manifest: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_task_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def build_balanced_subset_manifest(
    manifest: Dict[str, Any],
    limits_by_subfamily: Dict[str, int],
    include_flag: str = "include_in_stage2",
) -> Dict[str, Any]:
    """
    Build a manifest copy that enables only the densest tasks from selected subfamilies.

    Tasks are ranked by valid_count descending within each subfamily.
    """
    manifest_copy = json.loads(json.dumps(manifest))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in manifest_copy["tasks"]:
        grouped.setdefault(task["subfamily"], []).append(task)

    keep = set()
    for subfamily, limit in limits_by_subfamily.items():
        tasks = sorted(grouped.get(subfamily, []), key=lambda t: t["valid_count"], reverse=True)
        keep.update(task["task_name"] for task in tasks[:limit])

    for task in manifest_copy["tasks"]:
        task["include_in_hpo"] = task["task_name"] in keep
        task["include_in_stage2"] = task["task_name"] in keep

    manifest_copy["subset_summary"] = {
        "include_flag": include_flag,
        "limits_by_subfamily": limits_by_subfamily,
        "selected_tasks": len(keep),
    }
    return recompute_family_summary(manifest_copy)


def build_hpo_subset_manifest(
    manifest: Dict[str, Any],
    limits_by_subfamily: Dict[str, int],
) -> Dict[str, Any]:
    """
    Build a manifest copy that narrows only the HPO subset.

    `include_in_stage2` is preserved so the full manifest can still drive the
    promoted Stage 2 training run.
    """
    manifest_copy = json.loads(json.dumps(manifest))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in manifest_copy["tasks"]:
        grouped.setdefault(task["subfamily"], []).append(task)

    keep = set()
    for subfamily, limit in limits_by_subfamily.items():
        tasks = sorted(grouped.get(subfamily, []), key=lambda t: t["valid_count"], reverse=True)
        keep.update(task["task_name"] for task in tasks[:limit])

    for task in manifest_copy["tasks"]:
        task["include_in_hpo"] = task["task_name"] in keep

    manifest_copy["subset_summary"] = {
        "include_flag": "include_in_hpo",
        "limits_by_subfamily": limits_by_subfamily,
        "selected_tasks": len(keep),
    }
    return recompute_family_summary(manifest_copy)


def select_label_cols(manifest: Dict[str, Any], include_flag: str = "include_in_stage2") -> List[str]:
    """Return selected task names from a task manifest."""
    return [task["task_name"] for task in manifest["tasks"] if task.get(include_flag, False)]


def selected_task_types(manifest: Dict[str, Any], include_flag: str = "include_in_stage2") -> Counter:
    """Return task-type counts for the selected subset."""
    counts: Counter = Counter()
    for task in manifest["tasks"]:
        if task.get(include_flag, False):
            counts[task["task_type"]] += 1
    return counts


def summarize_manifest(manifest: Dict[str, Any]) -> str:
    """Create a short human-readable summary for logs."""
    parts = [f"n_tasks={manifest['n_tasks']}"]
    for family, summary in sorted(manifest.get("family_summary", {}).items()):
        parts.append(
            f"{family}:tasks={summary['n_tasks']},"
            f"hpo={summary['include_in_hpo']},"
            f"stage2={summary['include_in_stage2']},"
            f"types={summary['task_types']}"
        )
    return " | ".join(parts)
