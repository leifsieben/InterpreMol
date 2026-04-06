from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def run_chemprop_chemeleon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_names: List[str],
    task_type: str,
    output_dir: str | Path,
    chemprop_env: str,
    freeze_encoder: bool,
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 5,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    train_df[["smiles", *task_names]].to_csv(train_csv, index=False)
    val_df[["smiles", *task_names]].to_csv(val_csv, index=False)
    test_df[["smiles", *task_names]].to_csv(test_csv, index=False)

    runner = Path(__file__).resolve().parent / "chemeleon_transfer_runner.py"
    command = [
        "bash",
        "-lc",
        " ".join(
            [
                f"source {chemprop_env}/bin/activate",
                "&&",
                "python",
                str(runner),
                f"--train-csv {train_csv}",
                f"--val-csv {val_csv}",
                f"--test-csv {test_csv}",
                f"--output-dir {output_dir}",
                f"--task-type {task_type}",
                "--freeze-encoder" if freeze_encoder else "",
                f"--batch-size {batch_size}",
                f"--epochs {epochs}",
                f"--patience {patience}",
            ]
        ),
    ]
    subprocess.run(command, check=True)

    result_path = output_dir / "result.json"
    with open(result_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload
