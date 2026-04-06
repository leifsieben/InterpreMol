from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model import InterpreMol, MLPHead  # noqa: E402


@dataclass
class TransferResult:
    model_name: str
    primary_metric: float
    metrics: Dict[str, float]
    predictions_path: str


class MoleculeFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, task_names: List[str]):
        self.task_names = task_names
        self.rows = []
        for _, row in frame.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            labels = row[task_names].to_numpy(dtype=np.float32)
            self.rows.append((mol, row["smiles"], labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        mol, smiles, labels = self.rows[idx]
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask_t = ~torch.isnan(labels_t)
        return mol, smiles, labels_t, mask_t


def collate_batch(batch):
    mols, smiles, labels, masks = zip(*batch)
    return list(mols), list(smiles), torch.stack(labels), torch.stack(masks)


class DownstreamInterpreMol(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, out_dim: int, head_hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = MLPHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            depth=2,
            out_dim=out_dim,
        )

    def forward(self, mols):
        x = self.encoder(mols)
        return self.head(x)


def masked_bce_loss(logits, labels, mask):
    if mask.sum() == 0:
        return logits.sum() * 0.0
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_clean, reduction="none")
    return (loss * mask.float()).sum() / mask.sum()


def masked_mse_loss(preds, labels, mask):
    if mask.sum() == 0:
        return preds.sum() * 0.0
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))
    loss = nn.functional.mse_loss(preds, labels_clean, reduction="none")
    return (loss * mask.float()).sum() / mask.sum()


def classification_metric(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    task_scores = []
    for task_idx in range(y_true.shape[1]):
        valid = mask[:, task_idx].astype(bool)
        if valid.sum() == 0:
            continue
        target = y_true[valid, task_idx]
        if len(np.unique(target)) < 2:
            continue
        pred = y_pred[valid, task_idx]
        task_scores.append(float(roc_auc_score(target, pred)))
    if not task_scores:
        raise RuntimeError("No classification tasks had enough valid labels/classes for ROC-AUC.")
    return {"roc_auc": float(np.mean(task_scores)), "task_count": float(len(task_scores))}


def regression_metric(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    task_scores = []
    for task_idx in range(y_true.shape[1]):
        valid = mask[:, task_idx].astype(bool)
        if valid.sum() == 0:
            continue
        target = y_true[valid, task_idx]
        pred = y_pred[valid, task_idx]
        task_scores.append(float(math.sqrt(mean_squared_error(target, pred))))
    if not task_scores:
        raise RuntimeError("No regression tasks had enough valid labels for RMSE.")
    return {"rmse": float(np.mean(task_scores)), "task_count": float(len(task_scores))}


def evaluate_predictions(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if task_type == "classification":
        return classification_metric(y_true=y_true, y_pred=y_pred, mask=mask)
    if task_type == "regression":
        return regression_metric(y_true=y_true, y_pred=y_pred, mask=mask)
    raise ValueError(f"Unsupported task_type '{task_type}'")


def _validation_objective(task_type: str, metrics: Dict[str, float]) -> float:
    if task_type == "classification":
        return metrics["roc_auc"]
    return -metrics["rmse"]


def _predict(model, loader, device: str, task_type: str):
    model.eval()
    smiles_all, y_true, y_pred, masks = [], [], [], []
    with torch.no_grad():
        for mols, smiles, labels, mask in loader:
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(mols)
            if task_type == "classification":
                preds = torch.sigmoid(logits)
            else:
                preds = logits
            smiles_all.extend(smiles)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            masks.append(mask.cpu().numpy())
    return (
        smiles_all,
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        np.concatenate(masks, axis=0),
    )


def _write_predictions(path: Path, task_names: List[str], smiles: List[str], y_true, y_pred, mask):
    records = []
    for idx, smi in enumerate(smiles):
        row = {"smiles": smi}
        for task_idx, task_name in enumerate(task_names):
            row[f"{task_name}__true"] = float(y_true[idx, task_idx]) if mask[idx, task_idx] else np.nan
            row[f"{task_name}__pred"] = float(y_pred[idx, task_idx])
        records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(path, index=False)


def run_transfer(
    checkpoint_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_names: List[str],
    task_type: str,
    output_dir: str | Path,
    freeze_encoder: bool,
    device: str = "cuda",
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 5,
    head_lr: float = 1e-3,
    encoder_lr: float = 5e-5,
    weight_decay: float = 1e-5,
) -> TransferResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = InterpreMol.load(checkpoint_path, device=device)
    config = pretrained.config
    model = DownstreamInterpreMol(
        encoder=copy.deepcopy(pretrained.encoder),
        d_model=int(config["d_model"]),
        out_dim=len(task_names),
        head_hidden_dim=int(config.get("mlp_hidden_dim", config["d_model"] * 2)),
    ).to(device)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
        model_name = "interpremol_frozen"
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        model_name = "interpremol_finetune"

    train_loader = DataLoader(
        MoleculeFrameDataset(train_df, task_names),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        MoleculeFrameDataset(val_df, task_names),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        MoleculeFrameDataset(test_df, task_names),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    best_state = None
    best_score = -float("inf")
    best_metrics = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for mols, _smiles, labels, mask in train_loader:
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(mols)
            if task_type == "classification":
                loss = masked_bce_loss(logits, labels, mask)
            else:
                loss = masked_mse_loss(logits, labels, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        _, val_y_true, val_y_pred, val_mask = _predict(model, val_loader, device=device, task_type=task_type)
        val_metrics = evaluate_predictions(task_type, val_y_true, val_y_pred, val_mask)
        objective = _validation_objective(task_type, val_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                **val_metrics,
            }
        )

        if objective > best_score:
            best_score = objective
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} never produced a valid validation checkpoint.")

    model.load_state_dict(best_state)
    smiles, test_y_true, test_y_pred, test_mask = _predict(model, test_loader, device=device, task_type=task_type)
    test_metrics = evaluate_predictions(task_type, test_y_true, test_y_pred, test_mask)
    predictions_path = output_dir / f"{model_name}_predictions.csv"
    _write_predictions(predictions_path, task_names, smiles, test_y_true, test_y_pred, test_mask)

    with open(output_dir / f"{model_name}_history.json", "w", encoding="utf-8") as handle:
        json.dump({"validation_best": best_metrics, "history": history}, handle, indent=2)

    primary_key = "roc_auc" if task_type == "classification" else "rmse"
    return TransferResult(
        model_name=model_name,
        primary_metric=float(test_metrics[primary_key]),
        metrics=test_metrics,
        predictions_path=str(predictions_path),
    )
