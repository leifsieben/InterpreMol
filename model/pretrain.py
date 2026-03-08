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
    print(f"Loading data from {config['data_file']}...")

    smiles_col = config["smiles_col"]
    label_cols = config.get("label_cols")

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
            shuffle_buffer_size=config.get("shuffle_buffer_size", 1000),
        )

        config["n_tasks"] = n_tasks
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
        if epoch < config["warmup_epochs"]:
            return epoch / config["warmup_epochs"]
        else:
            progress = (epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
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
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    grad_accum_steps = max(1, int(config["grad_accum_steps"]))
    use_amp = bool(config.get("use_amp", False) and scaler is not None and str(device).startswith("cuda"))
    max_oom_skips = int(config.get("max_oom_skips_per_epoch", 20))
    accum_counter = 0
    skipped_no_label = 0
    skipped_oom = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad(set_to_none=True)

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

        total_loss += loss.item() * grad_accum_steps * batch_samples
        total_samples += batch_samples

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

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: str,
    config: Dict
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for mols, labels, masks in tqdm(val_loader, desc="Validating"):
        labels = labels.to(device)
        masks = masks.to(device)

        if config["use_amp"] and str(device).startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=True):
                preds = model(mols)
                loss = criterion(preds, labels, masks)
        else:
            preds = model(mols)
            loss = criterion(preds, labels, masks)

        batch_samples = masks.sum().item()
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples

    return total_loss / max(total_samples, 1)


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
    if config["loss"] == "multitask-bce":
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
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, config, scaler, epoch
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, config)
        val_losses.append(val_loss)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

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
                model, optimizer, epoch + 1, val_loss, config,
                is_best=is_best,
                extra={
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
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
        'config': config,
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
    search_space = {
        **config,
        # Keep trials within a stable region for single A10G GPUs.
        "use_amp": False,
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([4, 8, 16]),
        "grad_accum_steps": tune.choice([1, 2, 4]),
        "d_model": tune.choice([128, 256]),
        "n_layers": tune.choice([4, 6]),
        "n_heads": tune.choice([4, 8]),
        "dim_ff": tune.choice([256, 512]),
        "dropout": tune.uniform(0.0, 0.3),
        "mlp_hidden_dim": tune.choice([256, 512]),
        "mlp_head_depth": tune.choice([2, 3]),
        "use_edge_bias": tune.choice([True, False]),
        "max_distance": tune.choice([4, 6, 8]),
    }

    def trainable(trial_config):
        results = train(trial_config)
        tune.report(val_loss=results['best_val_loss'])

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=config["epochs"],
        grace_period=10,
        reduction_factor=2
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

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.data_file:
        config["data_file"] = args.data_file
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

    # Resolve important paths once so Ray workers get absolute paths.
    config["data_file"] = resolve_project_path(config["data_file"])
    config["checkpoint_dir"] = resolve_project_path(config["checkpoint_dir"])

    # Print config
    print("Configuration:")
    print(json.dumps(config, indent=2, default=str))

    # Run
    if args.hyperopt:
        best_config = hyperopt(config, args.num_samples)
        print(f"\nBest configuration saved to {config['checkpoint_dir']}/best_config.json")
    else:
        results = train(config, resume_path=args.resume)
        print(f"\nResults saved to {config['checkpoint_dir']}/results.json")


if __name__ == "__main__":
    main()
