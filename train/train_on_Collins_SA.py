import pandas as pd
from train import train_model, trainable, predict, MoleculeDataset
from model import InterpreMol
from rdkit import Chem
from ray import tune

# Load and prepare data
train_SA = pd.read_csv("Collins_SA_cleaned.csv")
smiles = train_SA["SMILES"].tolist()
labels = train_SA["Hit"].astype(float).tolist()

# Train/Validation split (80/20 split)
split = int(0.8 * len(smiles))
train_smiles, val_smiles = smiles[:split], smiles[split:]
train_labels, val_labels = labels[:split], labels[split:]

# Base configuration for non-tunable parameters and default values
base_config = {
    "loss": "bce",  # binary classification
    "batch_size": 32,
    "epochs": 100,  # epochs per trial
    "device": "cuda",  # use GPU if available
    "train_smiles": train_smiles,
    "train_labels": train_labels,
    "val_smiles": val_smiles,
    "val_labels": val_labels,
    "early_stopping_patience": 10,  # early stopping criteria
}

# Define a hyperparameter search space including model architecture parameters
search_space = {
    # Optimizer hyperparameters
    "lr": tune.loguniform(1e-5, 1e-3),
    "weight_decay": tune.loguniform(1e-6, 1e-2),

    # Architecture hyperparameters (expanded for larger models)
    "d_model": tune.choice([128, 256, 512]),
    "n_layers": tune.choice([4, 6, 8, 12]),
    "n_heads": tune.choice([4, 8, 16]),
    "dim_ff": tune.choice([256, 512, 1024]),
    "dropout": tune.uniform(0.0, 0.4),
    "mlp_hidden_dim": tune.choice([256, 512, 1024]),
    "mlp_head_depth": tune.choice([2, 3, 4]),

    # Edge bias hyperparameters
    "use_edge_bias": tune.choice([True, False]),
    "max_distance": tune.choice([4, 6, 8]),

    # CLS token
    "use_cls_token": True,

    # Include non-tunable parameters from base_config
    **base_config
}

# Run hyperparameter optimization using Ray Tune
analysis = tune.run(
    trainable,
    config=search_space,
    num_samples=30,  # Increase the number of samples for thorough exploration
    metric="val_loss",
    mode="min",
    resources_per_trial={"cpu": 4, "gpu": 1}  # Adjust as needed
)

# Retrieve the best configuration based on validation loss
best_config = analysis.get_best_config(metric="val_loss", mode="min")
print("Best hyperparameters found:", best_config)

# Update best_config with the complete data and set device for final training (cpu or gpu)
best_config["train_smiles"] = train_smiles
best_config["train_labels"] = train_labels
best_config["val_smiles"] = val_smiles
best_config["val_labels"] = val_labels
best_config["device"] = "cuda"  # change to "cpu" if no GPU

# Train final model using the best configuration
final_model, best_loss = train_model(best_config)
final_model.save("best_Collins_SA_InterpreMol_model.pt")
print(f"Final best validation loss: {best_loss}")

# (Optional) Predict on the full dataset and store the predictions in the DataFrame
full_dataset = MoleculeDataset(smiles, labels)
predictions = predict(final_model, full_dataset, batch_size=64, device=best_config["device"])
train_SA["Predicted_Prob"] = predictions

# Save the DataFrame with predictions to CSV if needed
train_SA.to_csv("Collins_SA_predictions.csv", index=False)
