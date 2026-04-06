from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-type", required=True, choices=["classification", "regression"])
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class CsvMolDataset(Dataset):
    def __init__(self, csv_path: str):
        frame = pd.read_csv(csv_path)
        self.task_names = [c for c in frame.columns if c != "smiles"]
        self.rows = []
        for _, row in frame.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            labels = row[self.task_names].to_numpy(dtype=np.float32)
            self.rows.append((row["smiles"], mol, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        smiles, mol, labels = self.rows[idx]
        labels_t = torch.tensor(labels, dtype=torch.float32)
        mask_t = ~torch.isnan(labels_t)
        return smiles, mol, labels_t, mask_t


def collate_batch(featurizer):
    def _fn(batch):
        smiles, mols, labels, masks = zip(*batch)
        mol_graphs = [featurizer(mol) for mol in mols]
        bmg = BatchMolGraph(mol_graphs)
        return list(smiles), bmg, torch.stack(labels), torch.stack(masks)

    return _fn


class ChemeleonTransferModel(nn.Module):
    def __init__(self, task_count: int, freeze_encoder: bool):
        super().__init__()
        model_path = Path.home() / ".chemprop" / "chemeleon_mp.pt"
        model_path.parent.mkdir(exist_ok=True)
        if not model_path.exists():
            urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", model_path)

        chemeleon_mp = torch.load(model_path, weights_only=True, map_location="cpu")
        self.encoder = BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        self.encoder.load_state_dict(chemeleon_mp["state_dict"])
        self.agg = MeanAggregation()
        hidden_dim = int(chemeleon_mp["hyper_parameters"]["d_h"])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, task_count),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, bmg: BatchMolGraph):
        bmg.to(next(self.parameters()).device)
        atom_repr = self.encoder(bmg)
        mol_repr = self.agg(atom_repr, bmg.batch)
        return self.head(mol_repr)


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


def evaluate_predictions(task_type, y_true, y_pred, mask):
    scores = []
    for task_idx in range(y_true.shape[1]):
        valid = mask[:, task_idx].astype(bool)
        if valid.sum() == 0:
            continue
        if task_type == "classification":
            target = y_true[valid, task_idx]
            if len(np.unique(target)) < 2:
                continue
            scores.append(float(roc_auc_score(target, y_pred[valid, task_idx])))
        else:
            scores.append(float(np.sqrt(mean_squared_error(y_true[valid, task_idx], y_pred[valid, task_idx]))))
    key = "roc_auc" if task_type == "classification" else "rmse"
    return {key: float(np.mean(scores)), "task_count": int(len(scores))}


def predict(model, loader, task_type, device):
    model.eval()
    smiles_all, y_true, y_pred, masks = [], [], [], []
    with torch.no_grad():
        for smiles, bmg, labels, mask in loader:
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(bmg)
            preds = torch.sigmoid(logits) if task_type == "classification" else logits
            smiles_all.extend(smiles)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            masks.append(mask.cpu().numpy())
    return smiles_all, np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(masks)


def write_predictions(path: Path, task_names, smiles, y_true, y_pred, mask):
    rows = []
    for i, smi in enumerate(smiles):
        row = {"smiles": smi}
        for j, task in enumerate(task_names):
            row[f"{task}__true"] = float(y_true[i, j]) if mask[i, j] else np.nan
            row[f"{task}__pred"] = float(y_pred[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    train_ds = CsvMolDataset(args.train_csv)
    val_ds = CsvMolDataset(args.val_csv)
    test_ds = CsvMolDataset(args.test_csv)
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    loader_args = {"batch_size": args.batch_size, "collate_fn": collate_batch(featurizer)}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = ChemeleonTransferModel(
        task_count=len(train_ds.task_names),
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    if args.freeze_encoder:
        params = model.head.parameters()
        model_name = "chemeleon_frozen"
    else:
        params = [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ]
        model_name = "chemeleon_finetune"

    optimizer = torch.optim.AdamW(params, lr=args.head_lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    best_metrics = None
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _smiles, bmg, labels, mask in train_loader:
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bmg)
            loss = masked_bce_loss(logits, labels, mask) if args.task_type == "classification" else masked_mse_loss(logits, labels, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        _, vy, vp, vm = predict(model, val_loader, args.task_type, device)
        metrics = evaluate_predictions(args.task_type, vy, vp, vm)
        objective = metrics["roc_auc"] if args.task_type == "classification" else -metrics["rmse"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if objective > best_score:
            best_score = objective
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    smiles, ty, tp, tm = predict(model, test_loader, args.task_type, device)
    test_metrics = evaluate_predictions(args.task_type, ty, tp, tm)
    pred_path = output_dir / f"{model_name}_predictions.csv"
    write_predictions(pred_path, train_ds.task_names, smiles, ty, tp, tm)
    summary = {
        "model_name": model_name,
        "metrics": test_metrics,
        "predictions_path": str(pred_path),
        "validation_best": best_metrics,
        "history": history,
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
