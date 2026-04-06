from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from benchmarks.chemprop_v2 import run_chemprop_chemeleon
from benchmarks.interpremol_transfer import run_transfer
from benchmarks.molnet import DATASETS, load_moleculenet_split, save_split_csvs
from benchmarks.tree_baselines import run_tree_baseline


DEFAULT_DATASETS = ["bbbp", "bace", "clintox", "hiv", "tox21", "esol", "freesolv", "lipo"]
DEFAULT_MODELS = [
    "interpremol_frozen",
    "chemeleon_frozen",
    "random_forest",
    "chemeleon_finetune",
    "interpremol_finetune",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MoleculeNet benchmarks on-node.")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 2 best_model.pt")
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--data-dir", default="benchmark_data")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=sorted(DATASETS))
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--chemprop-env", default="/home/ubuntu/venvs/chemprop2-py311")
    return parser.parse_args()


def run_one_dataset(
    args,
    dataset_name: str,
    split: str,
    seed: int,
    run_root: Path,
    summaries: List[dict],
):
    spec, task_names, train_df, val_df, test_df = load_moleculenet_split(
        dataset_name=dataset_name,
        split=split,
        seed=seed,
        data_dir=args.data_dir,
    )
    combo_dir = run_root / dataset_name / split / f"seed_{seed}"
    split_csvs = save_split_csvs(combo_dir / "splits", task_names, train_df, val_df, test_df)

    for model_name in args.models:
        model_dir = combo_dir / model_name
        if model_name == "interpremol_frozen":
            result = run_transfer(
                checkpoint_path=args.checkpoint,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                task_names=task_names,
                task_type=spec.task_type,
                output_dir=model_dir,
                freeze_encoder=True,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                head_lr=args.head_lr,
                encoder_lr=args.encoder_lr,
            )
            metrics = result.metrics
            predictions_path = result.predictions_path
        elif model_name == "interpremol_finetune":
            result = run_transfer(
                checkpoint_path=args.checkpoint,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                task_names=task_names,
                task_type=spec.task_type,
                output_dir=model_dir,
                freeze_encoder=False,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                head_lr=args.head_lr,
                encoder_lr=args.encoder_lr,
            )
            metrics = result.metrics
            predictions_path = result.predictions_path
        elif model_name in {"random_forest", "xgboost"}:
            payload = run_tree_baseline(
                model_name=model_name,
                train_df=pd.concat([train_df, val_df], ignore_index=True),
                test_df=test_df,
                task_names=task_names,
                task_type=spec.task_type,
                output_dir=model_dir,
            )
            metrics = payload["metrics"]
            predictions_path = payload["predictions_path"]
        elif model_name in {"chemeleon_frozen", "chemeleon_finetune"}:
            payload = run_chemprop_chemeleon(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                task_names=task_names,
                task_type=spec.task_type,
                output_dir=model_dir,
                chemprop_env=args.chemprop_env,
                freeze_encoder=(model_name == "chemeleon_frozen"),
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
            metrics = payload["metrics"]
            predictions_path = payload["predictions_path"]
        else:
            raise ValueError(f"Unsupported model '{model_name}'")

        summary = {
            "dataset": dataset_name,
            "split": split,
            "seed": seed,
            "task_type": spec.task_type,
            "model": model_name,
            "metric_name": spec.primary_metric,
            "metric_value": float(metrics[spec.primary_metric]),
            "task_count": int(metrics["task_count"]),
            "predictions_path": predictions_path,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_csv": split_csvs["train"],
            "val_csv": split_csvs["val"],
            "test_csv": split_csvs["test"],
        }
        summaries.append(summary)
        with open(model_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"moleculenet_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    with open(run_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    for dataset_name in args.datasets:
        for split in args.splits:
            for seed in args.seeds:
                run_one_dataset(args, dataset_name, split, seed, run_root, summaries)
                pd.DataFrame(summaries).to_csv(run_root / "summary.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        agg = (
            summary_df.groupby(["dataset", "split", "model", "metric_name"], as_index=False)["metric_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.to_csv(run_root / "aggregate.csv", index=False)


if __name__ == "__main__":
    main()
