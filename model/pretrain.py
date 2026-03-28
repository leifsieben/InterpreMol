#!/usr/bin/env python3
"""
InterpreMol Pretraining Script

Supports:
- Multi-task pretraining on large datasets
- Mixed precision training (AMP)
- Gradient accumulation
- Periodic checkpointing with S3 backup
- Resume from checkpoint
- Hyperparameter optimization with Ray Tune

Usage:
    # Single training run
    python pretrain.py --config config.json

    # Hyperparameter optimization
    python pretrain.py --config config.json --hyperopt --num-samples 30

    # Resume from checkpoint
    python pretrain.py --config config.json --resume checkpoints/latest
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from model import InterpreMol
from train import (
    MultiTaskMoleculeDataset,
    multitask_collate,
    masked_bce_loss,
    masked_mse_loss,
)
from aws_utils import S3Manager, CheckpointManager, get_instance_metadata
from task_manifest import (
    build_balanced_subset_manifest,
    build_hpo_subset_manifest,
    build_task_manifest,
    load_task_manifest,
    save_task_manifest,
    select_label_cols,
    selected_task_types,
    summarize_manifest,
)


def masked_multiclass_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss with masking for multiclass wide-table tasks.

    Args:
        logits: [batch_size, n_tasks, n_classes]
        labels: [batch_size, n_tasks]
        mask: [batch_size, n_tasks]
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0

    labels_clean = torch.where(mask, labels, torch.zeros_like(labels)).long()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels_clean.reshape(-1)
    flat_mask = mask.reshape(-1)

    valid_logits = flat_logits[flat_mask]
    valid_labels = flat_labels[flat_mask]
    return nn.functional.cross_entropy(valid_logits, valid_labels)


def _binary_baseline_loss(class_counts: Dict[str, int]) -> float:
    total = sum(class_counts.values())
    if total == 0:
        return 1.0
    pos = float(class_counts.get("1", 0))
    p = min(max(pos / total, 1e-6), 1 - 1e-6)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _multiclass_baseline_loss(class_counts: Dict[str, int]) -> float:
    total = sum(class_counts.values())
    if total == 0:
        return 1.0
    baseline = 0.0
    for count in class_counts.values():
        prob = count / total
        if prob > 0:
            baseline -= prob * math.log(prob)
    return float(max(baseline, 1e-6))


def build_task_groups_from_manifest(manifest: Dict[str, Any], include_flag: str) -> Dict[str, Any]:
    """Build typed task group metadata aligned to the selected label order."""
    selected_tasks = [task for task in manifest["tasks"] if task.get(include_flag, False)]
    task_to_index = {task["task_name"]: idx for idx, task in enumerate(selected_tasks)}

    groups: Dict[str, Any] = {"selected_label_cols": [task["task_name"] for task in selected_tasks]}
    grouped_tasks: Dict[str, List[Dict[str, Any]]] = {}

    for task in selected_tasks:
        group_name = f"{task['task_type']}::{task['broad_family']}"
        grouped_tasks.setdefault(group_name, []).append(task)

    for group_name, group_tasks in grouped_tasks.items():
        if group_tasks[0]["task_type"] == "binary":
            baseline_losses = [_binary_baseline_loss(task["class_counts"]) for task in group_tasks]
        elif group_tasks[0]["task_type"] == "multiclass":
            baseline_losses = [_multiclass_baseline_loss(task["class_counts"]) for task in group_tasks]
        else:
            baseline_losses = [1.0 for _ in group_tasks]

        groups[group_name] = {
            "label_cols": [task["task_name"] for task in group_tasks],
            "indices": [task_to_index[task["task_name"]] for task in group_tasks],
            "out_dim": len(group_tasks),
            "num_classes": int(group_tasks[0]["num_classes"]),
            "task_type": group_tasks[0]["task_type"],
            "broad_family": group_tasks[0]["broad_family"],
            "baseline_loss": float(sum(baseline_losses) / max(len(baseline_losses), 1)),
        }

    return groups


class TypedTaskCriterion:
    """Mixed-task criterion with family-aware normalized losses."""

    def __init__(self, task_groups: Dict[str, Any]):
        self.task_groups = task_groups

    def compute_components(self, preds, labels, masks) -> Dict[str, Any]:
        components: Dict[str, Any] = {"groups": {}}
        normalized_losses = []
        total_valid = 0

        for group_name, group_cfg in self.task_groups.items():
            if group_name == "selected_label_cols" or group_cfg["out_dim"] == 0:
                continue

            idx = torch.tensor(group_cfg["indices"], device=labels.device, dtype=torch.long)
            group_labels = labels.index_select(1, idx)
            group_masks = masks.index_select(1, idx)
            group_preds = preds[group_name]
            valid_count = int(group_masks.sum().item())

            if group_cfg["task_type"] == "binary":
                raw_loss = masked_bce_loss(group_preds, group_labels, group_masks)
            elif group_cfg["task_type"] == "multiclass":
                raw_loss = masked_multiclass_loss(group_preds, group_labels, group_masks)
            else:
                raise ValueError(f"Unsupported task group type: {group_cfg['task_type']}")

            baseline = max(float(group_cfg.get("baseline_loss", 1.0)), 1e-6)
            normalized_loss = raw_loss / baseline

            components["groups"][group_name] = {
                "raw_loss": raw_loss,
                "normalized_loss": normalized_loss,
                "valid_count": valid_count,
                "baseline_loss": baseline,
                "broad_family": group_cfg["broad_family"],
                "task_type": group_cfg["task_type"],
            }
            normalized_losses.append(normalized_loss)
            total_valid += valid_count

        if not normalized_losses:
            raise ValueError("No active typed-task groups were selected for training.")

        components["total_loss"] = sum(normalized_losses) / len(normalized_losses)
        components["total_valid"] = total_valid
        return components

    def __call__(self, preds, labels, masks):
        return self.compute_components(preds, labels, masks)["total_loss"]


def build_typed_criterion(config: Dict):
    """Create a mixed-task criterion from manifest-derived task groups."""
    task_groups = config.get("task_groups")
    if not task_groups:
        raise ValueError("Typed-task criterion requested without config['task_groups'].")
    return TypedTaskCriterion(task_groups)


def sanitize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ephemeral runtime-only config keys before serialization."""
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def get_required_val_groups(config: Dict[str, Any]) -> list[str]:
    """Return the task groups that must appear in validation for selection/HPO."""
    explicit = config.get("required_val_groups")
    if explicit:
        return list(explicit)

    task_groups = config.get("task_groups", {})
    return [
        group_name
        for group_name, group_cfg in task_groups.items()
        if group_name != "selected_label_cols" and group_cfg.get("out_dim", 0) > 0
    ]


def finalize_val_metrics(config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Attach validation coverage diagnostics and the guarded selection loss."""
    required_groups = get_required_val_groups(config)
    min_group_count = int(config.get("min_val_group_count", 1))
    enforce_coverage = bool(config.get("enforce_val_group_coverage", True))

    group_valid_counts = metrics.get("group_valid_counts", {})
    covered_groups = [
        group_name for group_name in required_groups
        if int(group_valid_counts.get(group_name, 0)) >= min_group_count
    ]
    missing_groups = [group_name for group_name in required_groups if group_name not in covered_groups]
    coverage_ratio = (
        float(len(covered_groups) / len(required_groups))
        if required_groups else 1.0
    )
    coverage_ok = (not required_groups) or not missing_groups

    selection_loss = metrics["loss"]
    if enforce_coverage and not coverage_ok:
        selection_loss = float("inf")

    metrics = dict(metrics)
    metrics.update({
        "required_groups": required_groups,
        "covered_groups": covered_groups,
        "missing_groups": missing_groups,
        "coverage_ratio": coverage_ratio,
        "coverage_ok": coverage_ok,
        "selection_loss": selection_loss,
    })
    return metrics


def resolve_project_path(path_str: str) -> str:
    """Resolve a possibly-relative path against project root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    # pretrain.py lives in <project_root>/model/
    project_root = Path(__file__).resolve().parent.parent
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return str(candidate)

    # Fallback to current working directory resolution.
    return str(path.resolve())


def get_default_config() -> Dict[str, Any]:
    """Default configuration for pretraining."""
    return {
        # Data
        "data_file": "datasets/all_datasets_fused_standardized.parquet",
        "smiles_col": "SMILES_std",
        "label_cols": None,  # Auto-detect
        "task_manifest": None,
        "task_manifest_include_flag": "include_in_stage2",
        "val_frac": 0.05,
        "seed": 42,

        # Model architecture
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "dim_ff": 512,
        "dropout": 0.1,
        "mlp_hidden_dim": 256,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "max_atoms": 192,

        # Training
        "batch_size": 64,
        "grad_accum_steps": 1,
        "max_oom_skips_per_epoch": 20,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "warmup_epochs": 5,
        "early_stopping_patience": 10,
        "loss": "multitask-bce",
        "family_loss_weights": {
            "binary": 1.0,
            "multiclass": 1.0,
        },

        # Mixed precision
        "use_amp": True,

        # Checkpointing
        "checkpoint_dir": "checkpoints",
        "checkpoint_every": 5,  # epochs
        "keep_last_n": 3,

        # S3 (optional)
        "s3_bucket": None,
        "s3_prefix": "interpremol",

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 0,
        "pin_memory": True,

        # Logging
        "log_every": 100,  # batches
        "max_train_batches_per_epoch": None,  # optional smoke-test limiter
        "max_val_batches": None,  # optional smoke-test limiter
        "required_val_groups": None,  # None -> require all active typed-task groups
        "min_val_group_count": 1,
        "enforce_val_group_coverage": True,

        # HPO scheduler
        "hpo_grace_period": None,  # None -> min(10, epochs)
        "hpo_reduction_factor": 2,
    }


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load config from JSON file, merging with defaults."""
    config = get_default_config()

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = json.load(f)
        config.update(user_config)

    return config


def setup_data(config: Dict) -> tuple:
    """Set up data loaders."""
    config["data_file"] = resolve_project_path(config["data_file"])
    if not str(config.get("device", "")).startswith("cuda") and config.get("pin_memory", True):
        config["pin_memory"] = False
    print(f"Loading data from {config['data_file']}...")

    smiles_col = config["smiles_col"]
    label_cols = config.get("label_cols")
    task_manifest_path = config.get("task_manifest")
    include_flag = config.get("task_manifest_include_flag", "include_in_stage2")

    if task_manifest_path:
        manifest = load_task_manifest(task_manifest_path)
        task_groups = build_task_groups_from_manifest(manifest, include_flag=include_flag)
        label_cols = task_groups["selected_label_cols"]
        config["label_cols"] = label_cols
        config["task_groups"] = task_groups
        config["task_heads"] = {
            group_name: {
                "out_dim": group_cfg["out_dim"],
                "num_classes": 1 if group_cfg["task_type"] == "binary" else group_cfg["num_classes"],
            }
            for group_name, group_cfg in task_groups.items()
            if group_name != "selected_label_cols" and group_cfg["out_dim"] > 0
        }
        config["task_manifest_summary"] = summarize_manifest(manifest)
        config["selected_task_types"] = dict(selected_task_types(manifest, include_flag=include_flag))
        print(f"Using task manifest {task_manifest_path}")
        print(f"Task manifest summary: {config['task_manifest_summary']}")
        print(f"Selected task types: {config['selected_task_types']}")

        if config["loss"] == "multitask-bce" and any(
            task_type != "binary" for task_type in config["selected_task_types"]
        ):
            raise ValueError(
                "Selected task manifest includes non-binary tasks, but config['loss'] is "
                "'multitask-bce'. Use a manifest subset that is purely binary or implement "
                "typed-task losses before launching pretraining."
            )

    # Use streaming for parquet files (memory efficient)
    if config["data_file"].endswith(".parquet") and config.get("streaming", True):
        from streaming_dataset import create_streaming_dataloaders

        train_loader, val_loader, n_tasks = create_streaming_dataloaders(
            config["data_file"],
            smiles_col=smiles_col,
            label_cols=label_cols,
            val_frac=config["val_frac"],
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 0),
            max_tasks=config.get("max_tasks"),
            max_atoms=config.get("max_atoms"),
            shuffle_buffer_size=config.get("shuffle_buffer_size", 1000),
            pin_memory=config.get("pin_memory", True),
        )

        config["n_tasks"] = n_tasks
        if not config.get("task_heads"):
            config["out_dim"] = n_tasks

        return train_loader, val_loader

    # Non-streaming fallback (loads all into memory)
    print("Using in-memory loading (set streaming=True for large files)")

    if config["data_file"].endswith(".parquet"):
        df = pd.read_parquet(config["data_file"])
    else:
        df = pd.read_csv(config["data_file"])

    # Filter label columns
    if label_cols is None:
        label_cols = [c for c in df.columns if "smiles" not in c.lower()]

    print(f"Found {len(df)} molecules, {len(label_cols)} tasks")

    # Create datasets
    train_ds, val_ds = MultiTaskMoleculeDataset.train_val_split(
        df,
        smiles_col=smiles_col,
        label_cols=label_cols,
        val_frac=config["val_frac"],
        seed=config["seed"]
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=multitask_collate,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=multitask_collate,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True)
    )

    config["n_tasks"] = train_ds.n_tasks
    if not config.get("task_heads"):
        config["out_dim"] = train_ds.n_tasks

    return train_loader, val_loader


def setup_model(config: Dict, device: str) -> tuple:
    """Set up model, optimizer, scheduler."""
    model = InterpreMol.from_config(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        warmup_epochs = max(int(config["warmup_epochs"]), 0)
        total_epochs = max(int(config["epochs"]), 1)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return epoch / warmup_epochs
        if total_epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return model, optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    config: Dict,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0
) -> Dict[str, Any]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0
    grad_accum_steps = max(1, int(config["grad_accum_steps"]))
    use_amp = bool(config.get("use_amp", False) and scaler is not None and str(device).startswith("cuda"))
    max_oom_skips = int(config.get("max_oom_skips_per_epoch", 20))
    accum_counter = 0
    skipped_no_label = 0
    skipped_oom = 0
    group_loss_sums: Dict[str, float] = {}
    group_valid_counts: Dict[str, int] = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad(set_to_none=True)
    max_train_batches = config.get("max_train_batches_per_epoch")

    def _optimizer_step():
        has_grad = any(p.grad is not None for p in model.parameters())
        if not has_grad:
            return
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (mols, labels, masks) in enumerate(pbar):
        if max_train_batches is not None and processed_batches >= int(max_train_batches):
            break
        labels = labels.to(device)
        masks = masks.to(device)
        batch_samples = int(masks.sum().item())

        # Skip fully unlabeled microbatches: they produce no useful gradient.
        if batch_samples == 0:
            skipped_no_label += 1
            continue

        try:
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    preds = model(mols)
                    loss = criterion(preds, labels, masks)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
            else:
                preds = model(mols)
                loss = criterion(preds, labels, masks)
                loss = loss / grad_accum_steps
                loss.backward()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            skipped_oom += 1
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if skipped_oom > max_oom_skips:
                raise RuntimeError(
                    f"Exceeded max OOM skips ({max_oom_skips}) in epoch {epoch + 1}. "
                    "Lower batch size/model size or increase grad_accum_steps."
                ) from exc
            continue

        accum_counter += 1
        if accum_counter % grad_accum_steps == 0:
            _optimizer_step()

        loss_value = loss.item() * grad_accum_steps
        total_loss += loss_value * batch_samples
        total_samples += batch_samples
        processed_batches += 1

        if hasattr(criterion, "compute_components"):
            with torch.no_grad():
                components = criterion.compute_components(preds, labels, masks)
            for group_name, group_metrics in components["groups"].items():
                valid_count = int(group_metrics["valid_count"])
                if valid_count == 0:
                    continue
                group_loss_sums[group_name] = group_loss_sums.get(group_name, 0.0) + (
                    float(group_metrics["normalized_loss"].item()) * valid_count
                )
                group_valid_counts[group_name] = group_valid_counts.get(group_name, 0) + valid_count

        if (batch_idx + 1) % config["log_every"] == 0:
            avg_loss = total_loss / max(total_samples, 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Flush tail gradients when number of microbatches isn't divisible by grad_accum_steps.
    if accum_counter % grad_accum_steps != 0:
        _optimizer_step()

    if skipped_no_label > 0 or skipped_oom > 0:
        print(
            f"Epoch {epoch + 1}: skipped {skipped_no_label} empty-label batches "
            f"and {skipped_oom} OOM batches"
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "processed_batches": processed_batches,
        "total_samples": total_samples,
        "group_losses": {
            group_name: group_loss_sums[group_name] / max(group_valid_counts[group_name], 1)
            for group_name in group_loss_sums
        },
        "group_valid_counts": group_valid_counts,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: str,
    config: Dict
) -> Dict[str, Any]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0
    max_val_batches = config.get("max_val_batches")
    skipped_no_label = 0
    group_loss_sums: Dict[str, float] = {}
    group_valid_counts: Dict[str, int] = {}

    for batch_idx, (mols, labels, masks) in enumerate(tqdm(val_loader, desc="Validating")):
        if max_val_batches is not None and processed_batches >= int(max_val_batches):
            break
        labels = labels.to(device)
        masks = masks.to(device)
        batch_samples = int(masks.sum().item())

        if batch_samples == 0:
            skipped_no_label += 1
            continue

        if config["use_amp"] and str(device).startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                preds = model(mols)
                loss = criterion(preds, labels, masks)
        else:
            preds = model(mols)
            loss = criterion(preds, labels, masks)

        total_loss += loss.item() * batch_samples
        total_samples += batch_samples
        processed_batches += 1

        if hasattr(criterion, "compute_components"):
            components = criterion.compute_components(preds, labels, masks)
            for group_name, group_metrics in components["groups"].items():
                valid_count = int(group_metrics["valid_count"])
                if valid_count == 0:
                    continue
                group_loss_sums[group_name] = group_loss_sums.get(group_name, 0.0) + (
                    float(group_metrics["normalized_loss"].item()) * valid_count
                )
                group_valid_counts[group_name] = group_valid_counts.get(group_name, 0) + valid_count

    if skipped_no_label > 0:
        print(f"Validation skipped {skipped_no_label} empty-label batches")

    return {
        "loss": total_loss / max(total_samples, 1),
        "processed_batches": processed_batches,
        "total_samples": total_samples,
        "group_losses": {
            group_name: group_loss_sums[group_name] / max(group_valid_counts[group_name], 1)
            for group_name in group_loss_sums
        },
        "group_valid_counts": group_valid_counts,
    }


def train(config: Dict, resume_path: Optional[str] = None) -> Dict:
    """
    Main training loop.

    Args:
        config: Training configuration
        resume_path: Optional path to checkpoint to resume from

    Returns:
        Results dict with final metrics
    """
    device = config["device"]
    saved_config = sanitize_runtime_config(config)
    print(f"Training on device: {device}")

    # Check if on EC2
    ec2_info = get_instance_metadata()
    if ec2_info.get('on_ec2'):
        print(f"Running on EC2: {ec2_info.get('instance_type')}")

    # Setup S3 if configured
    s3_manager = None
    if config.get("s3_bucket"):
        s3_manager = S3Manager(config["s3_bucket"], config.get("s3_prefix", "interpremol"))
        print(f"S3 backup enabled: s3://{config['s3_bucket']}/{config.get('s3_prefix', 'interpremol')}")

    # Setup checkpoint manager
    ckpt_manager = CheckpointManager(
        config["checkpoint_dir"],
        s3_manager=s3_manager,
        keep_last_n=config["keep_last_n"]
    )

    # Setup data
    train_loader, val_loader = setup_data(config)

    # Setup model
    model, optimizer, scheduler = setup_model(config, device)

    # Setup loss
    if config["loss"] == "typed-multitask":
        criterion = build_typed_criterion(config)
    elif config["loss"] == "multitask-bce":
        criterion = masked_bce_loss
    elif config["loss"] == "multitask-mse":
        criterion = masked_mse_loss
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")

    # Setup AMP
    scaler = GradScaler("cuda") if config["use_amp"] and str(device).startswith("cuda") else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path:
        print(f"Resuming from {resume_path}...")
        ckpt = ckpt_manager.load_checkpoint(resume_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float("inf"))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Training loop
    train_losses = []
    val_losses = []
    val_selection_losses = []
    train_group_history = []
    val_group_history = []
    val_coverage_history = []
    epochs_without_improvement = 0
    report_fn = config.get("_report_fn")

    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, config, scaler, epoch
        )
        train_loss = train_metrics["loss"]
        train_losses.append(train_loss)
        train_group_history.append(train_metrics["group_losses"])

        # Validate
        val_metrics = finalize_val_metrics(
            config,
            validate(model, val_loader, criterion, device, config),
        )
        raw_val_loss = val_metrics["loss"]
        val_loss = val_metrics["selection_loss"]
        val_losses.append(raw_val_loss)
        val_selection_losses.append(val_loss)
        val_group_history.append(val_metrics["group_losses"])
        val_coverage_history.append({
            "coverage_ratio": val_metrics["coverage_ratio"],
            "coverage_ok": val_metrics["coverage_ok"],
            "missing_groups": val_metrics["missing_groups"],
            "group_valid_counts": val_metrics["group_valid_counts"],
        })

        # Step scheduler only if at least one optimizer step likely occurred.
        if train_metrics["processed_batches"] > 0:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss(raw/selection): {raw_val_loss:.4f}/{val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        if train_metrics["group_losses"] or val_metrics["group_losses"]:
            print(f"Train groups: {train_metrics['group_losses']}")
            print(f"Val groups: {val_metrics['group_losses']}")
        if val_metrics["required_groups"]:
            print(
                f"Val coverage: {len(val_metrics['covered_groups'])}/{len(val_metrics['required_groups'])} "
                f"groups (min_count={config.get('min_val_group_count', 1)})"
            )
            if not val_metrics["coverage_ok"]:
                print(f"Missing val groups: {val_metrics['missing_groups']}")

        if report_fn is not None:
            report_fn({
                "training_iteration": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_loss_raw": raw_val_loss,
                "lr": current_lr,
                "val_coverage_ratio": val_metrics["coverage_ratio"],
                "val_coverage_ok": float(val_metrics["coverage_ok"]),
                **{f"train_{k}": v for k, v in train_metrics["group_losses"].items()},
                **{f"val_{k}": v for k, v in val_metrics["group_losses"].items()},
                **{f"val_count_{k}": v for k, v in val_metrics["group_valid_counts"].items()},
            })

        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        if (epoch + 1) % config["checkpoint_every"] == 0 or is_best:
            ckpt_manager.save_checkpoint(
                model, optimizer, epoch + 1, val_loss, saved_config,
                is_best=is_best,
                extra={
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_selection_losses': val_selection_losses,
                    'train_group_history': train_group_history,
                    'val_group_history': val_group_history,
                    'val_coverage_history': val_coverage_history,
                }
            )
            print(f"Checkpoint saved (is_best={is_best})")

        # Early stopping
        patience = config.get("early_stopping_patience")
        if patience and epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

    # Save final results
    results = {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_selection_losses': val_selection_losses,
        'train_group_history': train_group_history,
        'val_group_history': val_group_history,
        'val_coverage_history': val_coverage_history,
        'config': saved_config,
        'n_tasks': config.get('n_tasks'),
        'timestamp': datetime.now().isoformat(),
    }

    ckpt_manager.save_results(results)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    return results


def hyperopt(config: Dict, num_samples: int = 30) -> Dict:
    """
    Run hyperparameter optimization with Ray Tune.

    Args:
        config: Base configuration
        num_samples: Number of trials

    Returns:
        Best configuration found
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    # Define search space
    architecture_choices = [
        {"d_model": 192, "n_layers": 4},
        {"d_model": 192, "n_layers": 6},
        {"d_model": 192, "n_layers": 8},
        {"d_model": 256, "n_layers": 4},
        {"d_model": 256, "n_layers": 6},
        {"d_model": 256, "n_layers": 8},
        {"d_model": 384, "n_layers": 4},
        {"d_model": 384, "n_layers": 6},
        {"d_model": 384, "n_layers": 8},
        {"d_model": 512, "n_layers": 4},
        {"d_model": 512, "n_layers": 6},
    ]
    search_space = {
        **config,
        # Keep trials within a stable region for single A10G GPUs.
        "use_amp": config.get("use_amp", True),
        "architecture": tune.choice(architecture_choices),
        "lr": tune.loguniform(3e-5, 3e-4),
        "weight_decay": tune.loguniform(1e-6, 3e-4),
        "batch_size": tune.choice([4, 8, 16]),
        "grad_accum_steps": tune.choice([1, 2, 4]),
        "n_heads": 8,
        "dropout": tune.uniform(0.05, 0.20),
        "mlp_head_depth": 2,
        "use_edge_bias": True,
        "max_distance": tune.choice([4, 6, 8]),
    }

    def trainable(trial_config):
        trial_config = dict(trial_config)
        architecture = trial_config.pop("architecture")
        trial_config.update(architecture)
        trial_config["dim_ff"] = 2 * trial_config["d_model"]
        trial_config["mlp_hidden_dim"] = 2 * trial_config["d_model"]

        def _report(metrics):
            try:
                tune.report(metrics)
            except TypeError:
                tune.report(**metrics)

        trial_config["_report_fn"] = _report
        results = train(trial_config)
        if results["best_val_loss"] == float("inf"):
            _report({"training_iteration": results["final_epoch"], "val_loss": float("inf")})

    max_t = int(config["epochs"])
    grace_cfg = config.get("hpo_grace_period")
    if grace_cfg is None:
        grace_period = min(10, max_t)
    else:
        grace_period = max(1, min(int(grace_cfg), max_t))
    reduction_factor = int(config.get("hpo_reduction_factor", 2))

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_t,
        grace_period=grace_period,
        reduction_factor=reduction_factor
    )

    run_kwargs = {
        "run_or_experiment": trainable,
        "config": search_space,
        "num_samples": num_samples,
        "scheduler": scheduler,
        "resources_per_trial": {
            "cpu": int(config.get("trial_cpus", 4)),
            "gpu": float(config.get("trial_gpus", 1)),
        },
        "name": "interpremol_hyperopt",
        "storage_path": str(Path(config["checkpoint_dir"]).resolve()),
    }

    # Ray >= 2.9 uses storage_path, while older versions still expect local_dir.
    try:
        analysis = tune.run(**run_kwargs)
    except TypeError as exc:
        if "storage_path" not in str(exc):
            raise
        run_kwargs.pop("storage_path", None)
        run_kwargs["local_dir"] = config["checkpoint_dir"]
        analysis = tune.run(**run_kwargs)

    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    architecture = best_config.pop("architecture")
    best_config.update(architecture)
    best_config["dim_ff"] = 2 * best_config["d_model"]
    best_config["mlp_hidden_dim"] = 2 * best_config["d_model"]
    print(f"Best config: {best_config}")

    # Save best config
    with open(Path(config["checkpoint_dir"]) / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)

    return best_config


def main():
    parser = argparse.ArgumentParser(description="InterpreMol Pretraining")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--hyperopt", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of hyperopt trials")
    parser.add_argument("--task-manifest", type=str, help="Path to a task manifest JSON file")
    parser.add_argument("--write-task-manifest", type=str, help="Write a task manifest JSON file before training")
    parser.add_argument("--write-balanced-subset-manifest", type=str, help="Write a balanced subset task manifest JSON file")
    parser.add_argument("--write-hpo-subset-manifest", type=str, help="Write an HPO-only subset task manifest JSON file")
    parser.add_argument("--subset-wong", type=int, help="Number of Wong_fused tasks to keep in balanced subset manifest")
    parser.add_argument("--subset-pcba", type=int, help="Number of PCBA_1328 tasks to keep in balanced subset manifest")
    parser.add_argument("--subset-l1000-mcf7", type=int, help="Number of L1000_MCF7 tasks to keep in balanced subset manifest")
    parser.add_argument("--subset-l1000-vcap", type=int, help="Number of L1000_VCAP tasks to keep in balanced subset manifest")
    parser.add_argument("--audit-only", action="store_true", help="Only build/write task manifest and exit")

    # Override config options from command line
    parser.add_argument("--data-file", type=str, help="Path to data file")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket for checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming for parquet")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks (for memory efficiency)")
    parser.add_argument("--max-atoms", type=int, help="Skip molecules with more than this many atoms")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches per epoch (smoke test)")
    parser.add_argument("--max-val-batches", type=int, help="Limit val batches per epoch (smoke test)")
    parser.add_argument("--hpo-grace-period", type=int, help="ASHA grace period for HPO")
    parser.add_argument("--hpo-reduction-factor", type=int, help="ASHA reduction factor for HPO")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.data_file:
        config["data_file"] = args.data_file
    if args.task_manifest:
        config["task_manifest"] = args.task_manifest
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        config["grad_accum_steps"] = args.gradient_accumulation
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.device:
        config["device"] = args.device
    if args.s3_bucket:
        config["s3_bucket"] = args.s3_bucket
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.no_streaming:
        config["streaming"] = False
    if args.max_tasks:
        config["max_tasks"] = args.max_tasks
    if args.max_atoms is not None:
        config["max_atoms"] = args.max_atoms
    if args.max_train_batches is not None:
        config["max_train_batches_per_epoch"] = args.max_train_batches
    if args.max_val_batches is not None:
        config["max_val_batches"] = args.max_val_batches
    if args.hpo_grace_period is not None:
        config["hpo_grace_period"] = args.hpo_grace_period
    if args.hpo_reduction_factor is not None:
        config["hpo_reduction_factor"] = args.hpo_reduction_factor

    # Resolve important paths once so Ray workers get absolute paths.
    config["data_file"] = resolve_project_path(config["data_file"])
    config["checkpoint_dir"] = resolve_project_path(config["checkpoint_dir"])
    if config.get("task_manifest"):
        config["task_manifest"] = resolve_project_path(config["task_manifest"])

    # Print config
    print("Configuration:")
    print(json.dumps(config, indent=2, default=str))

    if args.write_task_manifest:
        task_manifest_path = resolve_project_path(args.write_task_manifest)
        manifest = build_task_manifest(config["data_file"], smiles_col=config["smiles_col"])
        save_task_manifest(manifest, task_manifest_path)
        print(f"\nTask manifest written to {task_manifest_path}")
        print(summarize_manifest(manifest))
        if args.audit_only:
            return

    if args.write_balanced_subset_manifest:
        if not config.get("task_manifest"):
            raise ValueError("--write-balanced-subset-manifest requires --task-manifest")
        base_manifest = load_task_manifest(config["task_manifest"])
        limits = {
            "Wong_fused": int(args.subset_wong or 0),
            "PCBA_1328": int(args.subset_pcba or 0),
            "L1000_MCF7": int(args.subset_l1000_mcf7 or 0),
            "L1000_VCAP": int(args.subset_l1000_vcap or 0),
        }
        subset_manifest = build_balanced_subset_manifest(base_manifest, limits)
        subset_path = resolve_project_path(args.write_balanced_subset_manifest)
        save_task_manifest(subset_manifest, subset_path)
        print(f"\nBalanced subset manifest written to {subset_path}")
        print(summarize_manifest(subset_manifest))
        if args.audit_only:
            return

    if args.write_hpo_subset_manifest:
        if not config.get("task_manifest"):
            raise ValueError("--write-hpo-subset-manifest requires --task-manifest")
        base_manifest = load_task_manifest(config["task_manifest"])
        limits = {
            "Wong_fused": int(args.subset_wong or 0),
            "PCBA_1328": int(args.subset_pcba or 0),
            "L1000_MCF7": int(args.subset_l1000_mcf7 or 0),
            "L1000_VCAP": int(args.subset_l1000_vcap or 0),
        }
        subset_manifest = build_hpo_subset_manifest(base_manifest, limits)
        subset_path = resolve_project_path(args.write_hpo_subset_manifest)
        save_task_manifest(subset_manifest, subset_path)
        print(f"\nHPO subset manifest written to {subset_path}")
        print(summarize_manifest(subset_manifest))
        if args.audit_only:
            return

    # Run
    if args.hyperopt:
        best_config = hyperopt(config, args.num_samples)
        print(f"\nBest configuration saved to {config['checkpoint_dir']}/best_config.json")
    else:
        results = train(config, resume_path=args.resume)
        print(f"\nResults saved to {config['checkpoint_dir']}/results.json")


if __name__ == "__main__":
    main()
