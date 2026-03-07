# InterpreMol

An interpretable molecular property prediction model combining Graph Transformers with Integrated Gradients attribution.

## Overview

InterpreMol is designed for:
- **Multi-task pretraining** on large molecular datasets (ChEMBL-scale)
- **Fine-tuning** on specific biological activity prediction tasks
- **Interpretability** via Integrated Gradients with reference baseline comparison

## Installation

```bash
cd /path/to/InterpreMol
python -m venv interpremol
source interpremol/bin/activate
pip install -r requirements.txt
```

## Architecture

### Edge-Biased Graph Transformer

Unlike standard transformers, InterpreMol incorporates molecular graph structure:

- **Edge-biased attention**: Attention scores are biased based on bond type (single, double, triple, aromatic) and shortest-path distance between atoms
- **Bond context features**: Each atom embedding includes information about its bonding environment
- **CLS token pooling**: A learned [CLS] token aggregates molecular representation

### Key Components

| Module | Description |
|--------|-------------|
| `atom_embedding.py` | Atom featurization with bond context |
| `edge_bias.py` | Edge bias encoder (bond types + graph distance) |
| `model.py` | InterpreMol model, GraphTransformerEncoder |
| `train.py` | Training loops, multi-task data loaders |
| `interpret.py` | Integrated Gradients with MCS-based reference baseline |

## Usage

### Multi-Task Pretraining

```python
from train import train_model_multitask

config = {
    # Data
    "data_file": "datasets/all_datasets_fused_standardized.parquet",
    "smiles_col": "SMILES_std",
    "val_frac": 0.1,

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
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 100,
    "early_stopping_patience": 10,
    "loss": "multitask-bce",
    "device": "cuda",
}

model, val_loss, logs = train_model_multitask(config)
model.save("pretrained_model.pt")
```

### Prediction

```python
from model import InterpreMol
from train import predict_smiles

model = InterpreMol.load("pretrained_model.pt")
preds = predict_smiles(model, ["CCO", "c1ccccc1"], classification=True)
# preds.shape: [2, n_tasks]
```

### Interpretability

```python
from interpret import interpret_smiles, interpret_comparison

# Basic interpretation (vs zero baseline)
img, scores = interpret_smiles("c1ccc(F)cc1", model)

# Compare analog to hit (SAR analysis)
img, scores = interpret_comparison(
    hit_smiles="c1ccccc1",        # benzene (reference)
    analog_smiles="c1ccc(F)cc1",  # fluorobenzene (query)
    model=model
)
# New atoms (F) are attributed against zero -> shows full impact
# Matched atoms are attributed against hit counterpart -> shows context change
```

## Data Format

### Multi-Task Wide Format

The data loader expects a wide-format table:

| SMILES | task_1 | task_2 | task_3 | ... |
|--------|--------|--------|--------|-----|
| CCO    | 1.0    | NaN    | 0.0    | ... |
| c1ccccc1 | 0.0  | 1.0    | NaN    | ... |

- One row per molecule
- One column per task
- Missing labels as NaN (handled via masked loss)
- Supports CSV and Parquet formats

## Configuration Reference

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding dimension |
| `n_layers` | 4 | Number of transformer layers |
| `n_heads` | 4 | Number of attention heads |
| `dim_ff` | 256 | Feed-forward hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `mlp_hidden_dim` | 256 | MLP head hidden dimension |
| `mlp_head_depth` | 2 | MLP head depth |
| `use_cls_token` | True | Use CLS token for pooling |
| `use_edge_bias` | True | Use edge-biased attention |
| `max_distance` | 6 | Max graph distance to encode |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `batch_size` | 32 | Batch size |
| `epochs` | 100 | Max epochs |
| `early_stopping_patience` | 10 | Early stopping patience |
| `loss` | "bce" | Loss type: bce, mse, multitask-bce, multitask-mse |

## Testing

```bash
source interpremol/bin/activate
cd model
python test_architecture.py
```

## Project Structure

```
InterpreMol/
├── model/
│   ├── atom_embedding.py    # Atom featurization
│   ├── edge_bias.py         # Edge-biased attention
│   ├── model.py             # InterpreMol model
│   ├── train.py             # Training & data loading
│   ├── interpret.py         # Integrated Gradients
│   └── test_architecture.py # Unit tests
├── train/
│   └── train_on_Collins_SA.py  # Hyperparameter optimization
├── datasets/
│   └── all_datasets_fused_standardized.parquet  # Pretraining data
├── requirements.txt
└── README.md
```

## References

- Integrated Gradients: Sundararajan et al., ICML 2017
- Graphormer (edge-biased attention): Ying et al., NeurIPS 2021
- MiniMol (pretraining approach): https://arxiv.org/abs/2404.14986
