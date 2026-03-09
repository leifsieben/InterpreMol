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

## Methodology 

### Hyperparameter Optimization 

We performed hyperparameter optimization as a two-stage procedure designed to reduce wall-clock time while preserving a principled model-selection process. In Stage 1 (fast screening), we ran a reduced-cost search on a single GPU using a limited trial budget (num_samples=12) and a short training horizon, with early stopping enabled and aggressive ASHA-based pruning. The search was intentionally proxy-based: each trial was allowed only a small training budget and additional runtime caps were imposed during HPO only (max_tasks=256, max_train_batches_per_epoch=250, max_val_batches=40) so that clearly underperforming configurations could be discarded early. Pruning was made deliberately aggressive by reducing the scheduler grace period to hpo_grace_period=2 and using hpo_reduction_factor=3, so trials had to show useful signal within the first few epochs to survive. This stage therefore served as a ranking mechanism rather than a final estimate of best achievable performance. The search space was the project’s predefined HPO parameter space in the training code, with the scheduler controls patched to expose the pruning hyperparameters explicitly; Stage 1 evaluated draws from that space under the shortened budget and produced a ranked list of candidate configurations.

In Stage 2 (refinement), we do not continue all Stage 1 trials. Instead, we take only the top 2–3 configurations from Stage 1, as ranked by the same validation objective used during HPO, and retrain them under fuller settings intended to better approximate final performance. Concretely, Stage 2 removes or relaxes the Stage 1 proxy caps and increases training fidelity by running longer (roughly 30–100 epochs) with less aggressive stopping (early_stopping_patience around 8–10). The decision rule from Stage 1 to Stage 2 is therefore: select the highest-ranked configurations under the Stage 1 validation metric, subject to keeping the finalist set small enough to make full retraining practical on 1 GPU. Operationally, the process was made repeatable by assigning each run a unique run ID, storing logs and heartbeat files, syncing artifacts to S3 on a schedule, and generating a stage2_template.json artifact that records how the finalist configurations should be instantiated for refinement. This gives a reproducible audit trail from initial screening through final candidate selection.
If you want, I can turn this into a tighter paper-style “Hyperparameter Optimization” subsection with explicit placeholders for the exact parameter names from your search space.

## References

- Integrated Gradients: Sundararajan et al., ICML 2017
- Graphormer (edge-biased attention): Ying et al., NeurIPS 2021
- MiniMol (pretraining approach): https://arxiv.org/abs/2404.14986
