import pandas as pd
from train import train_model, predict, MoleculeDataset
from model import InterpreMol
import torch
import os
import pickle
import matplotlib.pyplot as plt

def main():
    # Limit CPU threads for consistency
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Load and prepare data
    train_SA = pd.read_csv("data/Collins_SA_cleaned.csv")
    smiles = train_SA["SMILES"].tolist()
    labels = train_SA["Hit"].astype(float).tolist()

    # Train/Validation split (80/20 split)
    split = int(0.8 * len(smiles))
    train_smiles, val_smiles = smiles[:split], smiles[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Define a static configuration for a single training run
    config = {
        "lr": 1e-4,                # A good starting learning rate
        "weight_decay": 1e-4,      # Moderate regularization
        "d_model": 128,            # Model dimensionality
        "n_layers": 3,             # A modest number of transformer layers
        "n_heads": 4,              # Number of attention heads
        "dim_ff": 256,             # Feedforward network dimension
        "dropout": 0.1,            # Small dropout to avoid overfitting
        "mlp_hidden_dim": 256,     # Hidden dimension for MLP head
        "mlp_head_depth": 2,       # Depth of MLP head
        "loss": "bce",             # Binary cross-entropy loss for binary classification
        "batch_size": 32,          # Batch size for training
        "epochs": 50,             # Maximum epochs (early stopping will likely cut this short)
        "device": "cuda",          # Change to "cpu" if GPU is unavailable
        "train_smiles": train_smiles,
        "train_labels": train_labels,
        "val_smiles": val_smiles,
        "val_labels": val_labels,
        "early_stopping_patience": 10,  # Stop early if no improvement in 10 epochs
        "use_cls_token": True      # Whether to use a [CLS] token for the classifier model
    }
 
    # Train the model with the given configuration and capture the logs
    final_model, best_loss, logs = train_model(config)
    final_model.save("first_Collins_SA_InterpreMol_model.pt")
    print(f"Final best validation loss: {best_loss:.4f}")

    # Save logs for future plotting
    with open("training_logs.pkl", "wb") as f:
        pickle.dump(logs, f)
    print("Training logs saved to 'training_logs.pkl'.")

    # Optionally: Plot the training/validation loss curves using the logs
    # (You can comment out or move this block to another script if preferred)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(logs["train_losses"]) + 1)
    plt.plot(epochs, logs["train_losses"], label="Train Loss")
    plt.plot(epochs, logs["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epoch")
    plt.axvline(logs["best_epoch"], color='gray', linestyle='--', label=f"Best Epoch: {logs['best_epoch']}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Predict on all data and save predictions to CSV
    full_dataset = MoleculeDataset(smiles, labels)
    predictions = predict(final_model, full_dataset, batch_size=64, device=config["device"])
    train_SA["Predicted_Prob"] = predictions
    train_SA.to_csv("Collins_SA_predictions_initial_model.csv", index=False)

if __name__ == "__main__":
    main()
 