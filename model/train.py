import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from model import InterpreMol
from atom_embedding import AtomFeaturizer

class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float)
        return mol, label

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
    train_ds = MoleculeDataset(config["train_smiles"], config["train_labels"])
    val_ds = MoleculeDataset(config["val_smiles"], config["val_labels"])

    def mol_collate(batch):
        mols, labels = zip(*batch)
        return list(mols), torch.stack(labels)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=mol_collate)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=mol_collate)
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
