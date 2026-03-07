import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from model import InterpreMol
from atom_embedding import AtomFeaturizer


class MoleculeDataset(torch.utils.data.Dataset):
    """Single-task molecule dataset."""

    def __init__(self, smiles_list, labels):
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles[idx])
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {self.smiles[idx]}")
        label = torch.tensor([self.labels[idx]], dtype=torch.float)
        return mol, label


class MultiTaskMoleculeDataset(torch.utils.data.Dataset):
    """
    Multi-task molecule dataset for wide-format tables.

    Handles missing labels (NaN) which are common in multi-task datasets
    like ChEMBL where not all molecules are tested on all assays.

    Args:
        df: pandas DataFrame with SMILES column and label columns
        smiles_col: name of the SMILES column
        label_cols: list of label column names (if None, uses all columns except smiles_col)
        validate_smiles: if True, filter out invalid SMILES on init (slower but safer)
    """

    def __init__(self, df, smiles_col="SMILES", label_cols=None, validate_smiles=False):
        self.smiles_col = smiles_col

        if label_cols is None:
            label_cols = [c for c in df.columns if c != smiles_col]
        self.label_cols = label_cols
        self.n_tasks = len(label_cols)

        if validate_smiles:
            # Filter out invalid SMILES
            valid_mask = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)
            df = df[valid_mask].reset_index(drop=True)
            print(f"Kept {len(df)} / {len(valid_mask)} molecules with valid SMILES")

        self.smiles = df[smiles_col].tolist()
        # Convert labels to numpy array, preserving NaN
        self.labels = df[label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles[idx])
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {self.smiles[idx]}")

        # Labels as tensor (NaN preserved)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        # Mask: True where label is valid (not NaN)
        mask = ~torch.isnan(labels)

        return mol, labels, mask

    @classmethod
    def from_csv(cls, path, smiles_col="SMILES", label_cols=None, **kwargs):
        """Load from CSV file."""
        df = pd.read_csv(path)
        return cls(df, smiles_col=smiles_col, label_cols=label_cols, **kwargs)

    @classmethod
    def from_parquet(cls, path, smiles_col="SMILES", label_cols=None, **kwargs):
        """Load from Parquet file."""
        df = pd.read_parquet(path)
        return cls(df, smiles_col=smiles_col, label_cols=label_cols, **kwargs)

    @classmethod
    def from_file(cls, path, smiles_col="SMILES", label_cols=None, **kwargs):
        """Load from CSV or Parquet file (auto-detect by extension)."""
        if path.endswith(".parquet"):
            return cls.from_parquet(path, smiles_col=smiles_col, label_cols=label_cols, **kwargs)
        elif path.endswith(".csv"):
            return cls.from_csv(path, smiles_col=smiles_col, label_cols=label_cols, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")

    @classmethod
    def train_val_split(cls, df, smiles_col="SMILES", label_cols=None,
                        val_frac=0.2, seed=42, **kwargs):
        """
        Create train/val datasets with random split.

        Returns:
            (train_dataset, val_dataset)
        """
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_frac))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        train_ds = cls(train_df, smiles_col=smiles_col, label_cols=label_cols, **kwargs)
        val_ds = cls(val_df, smiles_col=smiles_col, label_cols=label_cols, **kwargs)

        return train_ds, val_ds


def multitask_collate(batch):
    """
    Collate function for multi-task datasets.

    Returns:
        mols: List[rdkit.Chem.Mol]
        labels: torch.Tensor [batch_size, n_tasks]
        masks: torch.Tensor [batch_size, n_tasks] (True = valid label)
    """
    mols, labels, masks = zip(*batch)
    return list(mols), torch.stack(labels), torch.stack(masks)


def masked_bce_loss(preds, labels, mask):
    """
    Binary cross-entropy loss with masking for missing labels.

    Args:
        preds: [batch_size, n_tasks] logits
        labels: [batch_size, n_tasks] targets (may contain NaN)
        mask: [batch_size, n_tasks] True where label is valid

    Returns:
        Scalar loss (mean over valid entries)
    """
    if mask.sum() == 0:
        # Keep loss attached to graph so AMP/GradScaler sees optimizer parameters.
        return preds.sum() * 0.0

    # Replace NaN with 0 to avoid NaN in loss computation
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))

    # Compute BCE for all entries
    bce = nn.functional.binary_cross_entropy_with_logits(
        preds, labels_clean, reduction='none'
    )

    # Apply mask and average
    masked_loss = (bce * mask.float()).sum() / mask.sum()
    return masked_loss


def masked_mse_loss(preds, labels, mask):
    """
    MSE loss with masking for missing labels.

    Args:
        preds: [batch_size, n_tasks] predictions
        labels: [batch_size, n_tasks] targets (may contain NaN)
        mask: [batch_size, n_tasks] True where label is valid

    Returns:
        Scalar loss (mean over valid entries)
    """
    if mask.sum() == 0:
        # Keep loss attached to graph so AMP/GradScaler sees optimizer parameters.
        return preds.sum() * 0.0

    # Replace NaN with 0
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))

    # Compute MSE for all entries
    mse = nn.functional.mse_loss(preds, labels_clean, reduction='none')

    # Apply mask and average
    masked_loss = (mse * mask.float()).sum() / mask.sum()
    return masked_loss

def get_loss(loss_type):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "multitask-bce":
        return nn.BCEWithLogitsLoss()  # for multi-task binary
    elif loss_type == "multitask-mse":
        return nn.MSELoss()  # for multi-task regression
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def get_dataloaders(config):
    """Get dataloaders for single-task training (legacy API)."""
    train_ds = MoleculeDataset(config["train_smiles"], config["train_labels"])
    val_ds = MoleculeDataset(config["val_smiles"], config["val_labels"])

    def mol_collate(batch):
        mols, labels = zip(*batch)
        return list(mols), torch.stack(labels)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=mol_collate)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=mol_collate)
    return train_loader, val_loader


def get_dataloaders_multitask(config):
    """
    Get dataloaders for multi-task training.

    Config should contain either:
        - train_df, val_df: pandas DataFrames
        - OR train_csv, val_csv: paths to CSV files
        - OR df with val_frac for automatic split

    Plus:
        - smiles_col: name of SMILES column (default: "SMILES")
        - label_cols: list of label columns (default: all except smiles_col)
        - batch_size: batch size
        - num_workers: dataloader workers (default: 0)
    """
    smiles_col = config.get("smiles_col", "SMILES")
    label_cols = config.get("label_cols", None)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 0)

    if "train_df" in config and "val_df" in config:
        train_ds = MultiTaskMoleculeDataset(
            config["train_df"], smiles_col=smiles_col, label_cols=label_cols
        )
        val_ds = MultiTaskMoleculeDataset(
            config["val_df"], smiles_col=smiles_col, label_cols=label_cols
        )
    elif "train_file" in config and "val_file" in config:
        # Auto-detect CSV or Parquet
        train_ds = MultiTaskMoleculeDataset.from_file(
            config["train_file"], smiles_col=smiles_col, label_cols=label_cols
        )
        val_ds = MultiTaskMoleculeDataset.from_file(
            config["val_file"], smiles_col=smiles_col, label_cols=label_cols
        )
    elif "train_csv" in config and "val_csv" in config:
        train_ds = MultiTaskMoleculeDataset.from_csv(
            config["train_csv"], smiles_col=smiles_col, label_cols=label_cols
        )
        val_ds = MultiTaskMoleculeDataset.from_csv(
            config["val_csv"], smiles_col=smiles_col, label_cols=label_cols
        )
    elif "data_file" in config:
        # Single file with automatic train/val split
        df = pd.read_parquet(config["data_file"]) if config["data_file"].endswith(".parquet") else pd.read_csv(config["data_file"])
        train_ds, val_ds = MultiTaskMoleculeDataset.train_val_split(
            df,
            smiles_col=smiles_col,
            label_cols=label_cols,
            val_frac=config.get("val_frac", 0.2),
            seed=config.get("seed", 42)
        )
    elif "df" in config:
        train_ds, val_ds = MultiTaskMoleculeDataset.train_val_split(
            config["df"],
            smiles_col=smiles_col,
            label_cols=label_cols,
            val_frac=config.get("val_frac", 0.2),
            seed=config.get("seed", 42)
        )
    else:
        raise ValueError("Config must contain train_df/val_df, train_file/val_file, train_csv/val_csv, data_file, or df")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=multitask_collate, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=multitask_collate, num_workers=num_workers
    )

    # Store n_tasks in config for model creation
    config["n_tasks"] = train_ds.n_tasks
    config["out_dim"] = train_ds.n_tasks

    return train_loader, val_loader

def train_model(config):
    device = config["device"]
    patience = config["early_stopping_patience"]
    if device == "cpu" and torch.cuda.is_available():
        print("⚠️ Warning: You're using CPU for training despite CUDA being available.")

    model = InterpreMol.from_config(config).to(device)
    print(f"Model initialized on device: {next(model.parameters()).device}")
    criterion = get_loss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-5)
    )

    train_loader, val_loader = get_dataloaders(config)

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    epochs_without_improvement = 0

    # Lists to track losses per epoch for logging
    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}", flush=True)
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        for batch_idx, (mols, labels) in enumerate(train_loader):
            labels = labels.to(device)
            preds = model(mols)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  [Batch {batch_idx+1}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for mols, labels in val_loader:
                labels = labels.to(device)
                preds = model(mols)
                epoch_val_loss += criterion(preds, labels).item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nBest validation loss of {best_val_loss:.4f} occurred at epoch {best_epoch}.", flush=True)

    # Restore best model state
    model.load_state_dict(best_model_state)

    # Return model, best validation loss, and the logs for further analysis
    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch
    }
    return model, best_val_loss, logs


def train_model_multitask(config):
    """
    Train a multi-task model with masked loss for missing labels.

    Config should contain:
        - Data: train_df/val_df, train_csv/val_csv, or df with val_frac
        - Model: d_model, n_layers, n_heads, dim_ff, dropout, mlp_hidden_dim, mlp_head_depth
        - Training: lr, weight_decay, epochs, batch_size, early_stopping_patience
        - loss: "multitask-bce" or "multitask-mse"
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    patience = config.get("early_stopping_patience", 10)

    # Get dataloaders (this also sets config["out_dim"])
    train_loader, val_loader = get_dataloaders_multitask(config)
    n_tasks = config["n_tasks"]
    print(f"Multi-task training with {n_tasks} tasks")

    # Create model
    model = InterpreMol.from_config(config).to(device)
    print(f"Model initialized on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Select loss function
    loss_type = config.get("loss", "multitask-bce")
    if loss_type == "multitask-bce":
        criterion = masked_bce_loss
    elif loss_type == "multitask-mse":
        criterion = masked_mse_loss
    else:
        raise ValueError(f"Unknown loss type for multi-task: {loss_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5)
    )

    # Learning rate scheduler (optional)
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config.get("lr_min", 1e-6)
        )

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for mols, labels, masks in pbar:
            labels = labels.to(device)
            masks = masks.to(device)

            preds = model(mols)
            loss = criterion(preds, labels, masks)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional)
            if config.get("grad_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()

            batch_valid = masks.sum().item()
            epoch_train_loss += loss.item() * batch_valid
            epoch_train_samples += batch_valid
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = epoch_train_loss / max(epoch_train_samples, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_samples = 0

        with torch.no_grad():
            for mols, labels, masks in val_loader:
                labels = labels.to(device)
                masks = masks.to(device)

                preds = model(mols)
                loss = criterion(preds, labels, masks)

                batch_valid = masks.sum().item()
                epoch_val_loss += loss.item() * batch_valid
                epoch_val_samples += batch_valid

        avg_val_loss = epoch_val_loss / max(epoch_val_samples, 1)
        val_losses.append(avg_val_loss)

        lr_str = f", LR: {scheduler.get_last_lr()[0]:.2e}" if scheduler else ""
        print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}{lr_str}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Restore best model
    model.load_state_dict(best_model_state)

    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "n_tasks": n_tasks
    }
    return model, best_val_loss, logs


def trainable(config):
    """Ray Tune compatible training function."""
    from ray import tune
    model, val_loss, _ = train_model(config)
    tune.report(val_loss=val_loss)


def trainable_multitask(config):
    """Ray Tune compatible multi-task training function."""
    from ray import tune
    model, val_loss, _ = train_model_multitask(config)
    tune.report(val_loss=val_loss)


def predict(model, dataset, batch_size=32, device=None, classification=True):
    """
    Predict labels using a trained model on a MoleculeDataset.

    Args:
        model (nn.Module): Trained InterpreMol model.
        dataset (torch.utils.data.Dataset): Dataset of (mol, label) pairs.
        batch_size (int): Batch size for prediction.
        device (str or torch.device): Device to use. If None, inferred from model.
        classification (bool): Whether to apply sigmoid for binary classification.

    Returns:
        List[float]: Predicted values (logits or probabilities).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: list(zip(*b))[0])
    predictions = []

    with torch.no_grad():
        for mols in loader:
            preds = model(mols).to("cpu")
            if classification:
                preds = torch.sigmoid(preds)
            predictions.extend(preds.squeeze().tolist())

    return predictions


def predict_multitask(model, dataset, batch_size=32, device=None, classification=True):
    """
    Predict labels using a trained multi-task model.

    Args:
        model: Trained InterpreMol model
        dataset: MultiTaskMoleculeDataset
        batch_size: Batch size for prediction
        device: Device to use (inferred from model if None)
        classification: Apply sigmoid for binary classification

    Returns:
        np.ndarray: Predictions [n_samples, n_tasks]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multitask_collate
    )

    all_preds = []

    with torch.no_grad():
        for mols, _, _ in loader:
            preds = model(mols).cpu()
            if classification:
                preds = torch.sigmoid(preds)
            all_preds.append(preds)

    return torch.cat(all_preds, dim=0).numpy()


def predict_smiles(model, smiles_list, batch_size=32, device=None, classification=True):
    """
    Predict directly from a list of SMILES strings.

    Args:
        model: Trained InterpreMol model
        smiles_list: List of SMILES strings
        batch_size: Batch size
        device: Device (inferred from model if None)
        classification: Apply sigmoid for binary classification

    Returns:
        np.ndarray: Predictions [n_samples, n_tasks]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        mols = []
        for smi in batch_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            mols.append(mol)

        with torch.no_grad():
            preds = model(mols).cpu()
            if classification:
                preds = torch.sigmoid(preds)
            all_preds.append(preds)

    result = torch.cat(all_preds, dim=0).numpy()
    return result
