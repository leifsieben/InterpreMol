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

### Phase 1: Valid Foundational Pretraining

The March 2026 pretraining run family should not be treated as authoritative. It demonstrated that the infrastructure runs, but the methodology was wrong for the fused table:

- the fused table is mixed-task, not purely binary
- `Wong_fused` and `PCBA_1328` are binary
- `L1000_MCF7` and `L1000_VCAP` are ternary with values `{0,1,2}`
- the old Stage 1 HPO used `multitask-bce` anyway
- the old Stage 1 HPO only searched the first `256` columns, which were `4 x Wong_fused + 252 x L1000_MCF7`
- Ray Tune only received one final report per trial, so ASHA pruning was not doing the intended job

The immediate goal is therefore Phase 1: produce one valid, fully trained foundational InterpreMol encoder from the fused supervised table.

Phase 1 starts with a task audit artifact. Every pretraining run must be driven by a manifest rather than by blindly taking every non-SMILES column. The manifest should record, for every task:

- task name
- broad family and subfamily
- task type
- number of classes
- valid label count
- value range
- class counts
- inclusion flags for HPO and Stage 2

`include_in_hpo` and `include_in_stage2` are intentionally separate. Stage 1 can use a dense proxy subset, while Stage 2 still retrains on the full validated task set.

The current expected family structure of `datasets/all_datasets_fused_standardized.parquet` is:

- `Wong_fused`: `4` binary tasks
- `PCBA_1328`: `1328` binary tasks
- `L1000_MCF7`: `978` ternary tasks
- `L1000_VCAP`: `978` ternary tasks

Pretraining must then use typed heads and typed losses:

- one binary multitask head for binary tasks with masked BCE-with-logits
- one ternary multitask head for L1000 tasks with masked cross-entropy
- a shared graph encoder underneath

The composite training and validation objective should be an equal-weight average over broad families so that the largest family does not dominate model selection. The recommended broad-family weighting is:

- `Wong_fused`
- `PCBA_1328`
- `L1000`, where `MCF7` and `VCAP` are averaged equally inside the family

### Phase 1 HPO Plan

The new HPO should still be two-stage, but actually defensible.

Stage 0 is a short smoke run on the typed-task setup. Its purpose is only to verify:

- manifest generation and loading
- binary and ternary losses both behave correctly
- no negative or non-finite losses occur
- per-family metrics are logged
- short validation windows are rejected if they do not cover all required task groups
- checkpoints and S3 backups work

Stage 1 is the proxy HPO stage. This stage should use intermediate validation reports so ASHA can prune aggressively. It should search a slightly wider architecture space than before, while reducing freedom elsewhere:

- `d_model`: `{192, 256, 384, 512}`
- `n_layers`: `{4, 6, 8}`
- `n_heads`: `8`
- `dim_ff = 2 * d_model`
- `mlp_hidden_dim = 2 * d_model`
- `mlp_head_depth = 2`
- `dropout`: search
- `lr`: search
- `weight_decay`: search
- `grad_accum_steps`: search
- `max_distance`: search
- `use_edge_bias = true`
- `use_cls_token = true`

To keep this wider search tractable, the expensive corner of the space should be constrained:

- if `d_model = 512`, only allow `n_layers in {4, 6}`
- if `n_layers = 8`, cap `d_model <= 384`

Stage 1 should remain aggressive:

- sequential single-GPU trials on a larger single node
- aggressive ASHA pruning based on repeated intermediate validation reports
- trial selection should use a guarded validation loss, not the raw loss, so incomplete validation coverage cannot win
- the proxy validation slice must be large enough to cover all required groups in practice; otherwise the guarded loss should stay invalid
- only the rank-1 configuration is promoted

Stage 2 is the actual foundational training run. It should retrain the single selected configuration on the full typed-task objective until convergence, preserving:

- `best_model.pt`
- latest checkpoint
- encoder-only artifact
- full config
- task manifest
- per-family metric history

Initial typed config templates for this workflow now live at:

- `configs/stage0_smoke_typed.json`
- `configs/hpo_stage1_typed.json`
- `configs/stage2_full_typed.json`

### Phase 1 Execution Record

The following items have actually been executed and should be treated as the current authoritative Phase 1 methodology record.

#### Task audit and manifest design

The fused supervised table is now handled through an explicit task manifest rather than by taking every non-SMILES column with one loss. The manifest split currently in use is:

- full Stage 2 task set: `3288` tasks
- binary tasks: `1332`
- multiclass tasks: `1956`
- family counts:
  - `Wong_fused`: `4`
  - `PCBA_1328`: `1328`
  - `L1000`: `1956`

The Stage 1 HPO proxy subset is drawn from the same fused table but uses only the tasks marked `include_in_hpo`:

- HPO proxy task set: `196` tasks
- binary tasks: `68`
- multiclass tasks: `128`
- family counts:
  - `Wong_fused`: `4`
  - `PCBA_1328`: `64`
  - `L1000`: `128`

This means the HPO runs train on the full fused dataset file, but optimize a typed proxy objective over a balanced task subset rather than all `3288` tasks directly.

#### Stage 0 smoke validation

Before HPO, the typed-task stack was smoke-tested locally and on the GPU node. These checks established:

- mixed binary + multiclass training runs end to end
- guarded validation rejects short windows that do not cover all required groups
- per-group train and validation losses are logged
- the old AMP configuration is unstable on the current A10G stack
- the operationally robust configuration on the current node is full precision with `use_amp = false`

This is why all later authoritative HPO runs use `use_amp = false`.

#### Stage 1 HPO round 1

Run:

- `hpo_stage1_typed_fp32_20260329_175407`

Configuration:

- `16` Ray Tune trials
- typed proxy objective over the `196`-task HPO subset
- architecture space:
  - `192 x {4,6,8}`
  - `256 x {4,6,8}`
  - `384 x {4,6,8}`
  - `512 x {4,6}`
- searched continuous/discrete knobs:
  - `lr`
  - `weight_decay`
  - `dropout`
  - `batch_size`
  - `grad_accum_steps`
  - `max_distance`
- ASHA pruning with intermediate validation reports
- guarded validation loss used as the selection metric

Observed behavior:

- all `16` trials completed successfully
- all `16` trials achieved full validation coverage
- all `16` trials showed large validation improvement from their first report
- best trial:
  - architecture: `512 x 6`
  - best validation loss: `0.4440`
- next best trials:
  - `384 x 6`: `0.4809`
  - `384 x 6`: `0.4851`
  - `256 x 8`: `0.5017`

Conclusion from round 1:

- the typed HPO pipeline is valid
- the proxy objective carries real learning signal
- strong candidates are concentrated in medium-to-large models, especially `512 x 6` and `384 x 6`

#### Stage 1 HPO round 2

Run:

- `hpo_stage1_typed_round2_20260330_115444`

Purpose:

- cast a wider architectural net without changing the typed proxy objective

Configuration:

- `32` Ray Tune trials
- same `196`-task typed HPO subset
- widened architecture space:
  - `128 x 4`
  - `128 x 6`
  - `192 x 10`
  - `256 x 10`
  - `384 x 10`
  - `512 x 8`
  - `640 x 4`
  - `640 x 6`
- widened ranges:
  - `lr: 2e-5 .. 5e-4`
  - `weight_decay: 1e-7 .. 5e-4`
  - `dropout: 0.03 .. 0.25`
  - `max_distance: {4,6,8,10}`

Observed behavior:

- all `32` trials completed successfully
- the widened space found additional viable regions
- best trial:
  - architecture: `192 x 10`
  - best validation loss: `0.4620`
- second best:
  - `640 x 4`: `0.4626`
- additional viable regions included:
  - `128 x 4`
  - `256 x 10`
  - `640 x 6`

Conclusion from round 2:

- the wider search discovered multiple new competitive regions
- however, round 2 did not beat the round 1 winner `512 x 6` with `0.4440`
- several new corners were clearly weak and mainly consumed budget

#### Phase 1 decision after rounds 1 and 2

The correct next move is not to widen the architecture net again. The broad search objective has already been achieved. A further pass should be a refinement pass around the empirically strong architecture families:

- `512 x 6`
- `384 x 6`
- `192 x 10`
- `640 x 4`
- optionally `256 x 10`

The next HPO pass should therefore keep the same typed proxy objective and refine:

- `lr`
- `weight_decay`
- `dropout`
- `max_distance`
- `batch_size`
- `grad_accum_steps`

The purpose of that refinement pass is to select a single final configuration for Stage 2 full-task pretraining on all `3288` tasks.

#### Stage 1 HPO round 3 refinement

Run:

- `hpo_stage1_typed_round3_refine_20260330_171107`

Purpose:

- refine around the empirically strong architecture families from rounds 1 and 2 rather than widening the net again

Configuration:

- `64` Ray Tune trials
- same `196`-task typed HPO subset
- refinement architecture space:
  - `512 x 6`
  - `384 x 6`
  - `192 x 10`
  - `640 x 4`
  - `256 x 10`
- refined ranges:
  - `lr: 3e-5 .. 4.5e-4`
  - `weight_decay: 1e-7 .. 1e-4`
  - `dropout: 0.06 .. 0.24`
  - `max_distance: {4,6,8,10}`

Observed behavior:

- all `64` trials completed successfully
- round 3 improved on rounds 2 in the refined regions
- best trial:
  - architecture: `256 x 10`
  - best validation loss: `0.4553`
- additional top trials:
  - `192 x 10`: `0.4521`
  - `256 x 10`: `0.4556`
  - `384 x 6`: `0.4570`

Conclusion from round 3:

- the refinement pass strengthened the case for the `192 x 10`, `256 x 10`, and `384 x 6` families
- however, it still did not beat the original round 1 winner `512 x 6`

#### Global HPO selection across all Stage 1 passes

The global best configuration over all completed HPO passes remains the round 1 winner:

- source run: `hpo_stage1_typed_fp32_20260329_175407`
- architecture: `512 x 6`
- best validation loss: `0.4440`

This is the configuration promoted into Stage 2.

#### Stage 2 full-task foundational pretraining

The first Stage 2 launch on April 1, 2026 was stopped immediately because it accidentally overrode some winning HPO optimization settings and therefore was not a strict promotion of the selected configuration.

The authoritative Stage 2 run is:

- `stage2_full_exactwinner_20260401_185331`

This launch uses:

- the exact global-best Stage 1 HPO architecture and optimization hyperparameters
- only the Stage 2 budget changes:
  - `task_manifest_include_flag = include_in_stage2`
  - `epochs = 40`
  - `warmup_epochs = 5`
  - `early_stopping_patience = 8`
  - `keep_last_n = 5`

The run trains on all validated tasks from the fused supervised table:

- `3288` total tasks
- `1332` binary tasks
- `1956` multiclass tasks

This is the current foundational encoder training run that should be monitored and preserved.

### Phase 2: MoleculeNet Benchmarking

Phase 2 starts only after Phase 1 produces a valid finalized foundational checkpoint. The benchmark should then compare:

- InterpreMol pretrained encoder with frozen encoder transfer
- InterpreMol pretrained encoder with end-to-end fine-tuning
- Chemprop v2 in its regular benchmark mode

The benchmark scope should cover the common MoleculeNet classification and regression tasks using both random and scaffold splits, aggregated over `3` seeds. This comparison is best-effort rather than fully co-tuned: we will use documented and reasonable settings for both methods, but we will not run a full per-dataset hyperparameter search for each benchmark model.

### Required Benchmark Outputs

For every MoleculeNet benchmark run, we should save:

- dataset name, split type, seed, and task type
- exact training config for InterpreMol and Chemprop v2
- pointer to the pretrained InterpreMol checkpoint used
- per-run metrics on validation and test sets
- per-molecule predictions on the test split
- aggregated summary tables over seeds
- generated plots

At minimum, the plotting set should include:

- per-dataset comparison bar plots for InterpreMol vs Chemprop v2
- random-split vs scaffold-split comparison plots
- seed-level variability plots or error bars
- training-curve plots for representative runs
- overall summary figure spanning all benchmark datasets

### Interpretability Validation Plan

After the benchmark pipeline is working, we need to verify that the integrated gradients interpretability workflow still behaves correctly with the final pretrained model and downstream fine-tuned models.

This validation should happen in two stages:

1. Technical validation
   - confirm the interpretability code still runs on the finalized model checkpoints
   - verify that attributions are stable enough across repeated runs and sensible baselines
   - confirm that atom-level attribution outputs align with the current model interface after pretraining and fine-tuning
2. Chemical sanity checking
   - work through real molecular examples rather than only synthetic smoke tests
   - compare attributed substructures with expected SAR-relevant motifs
   - identify cases where the explanation is chemically plausible and cases where it is not
   - capture representative successful and failure-case examples for later figures and discussion

## Action Items

1. Finalize Stage 2 training and preserve the selected pretrained model artifacts.
2. Create a reproducibility bundle containing the selected config, weights, checkpoints, logs, code revision, dataset identity, and S3 location.
3. Implement a dedicated MoleculeNet benchmark pipeline for InterpreMol and Chemprop v2.
4. Support both transfer modes for InterpreMol: full fine-tuning and frozen encoder.
5. Run both random and scaffold split evaluations across the selected MoleculeNet tasks with 3 seeds.
6. Save per-run metrics, per-split predictions, and aggregate benchmark tables.
7. Produce plots for every benchmark family and for the final aggregate comparison.
8. Revalidate integrated gradients on the finalized model checkpoints.
9. Run qualitative interpretability case studies on chemically meaningful examples and record both strong and weak cases.

## References

- Integrated Gradients: Sundararajan et al., ICML 2017
- Graphormer (edge-biased attention): Ying et al., NeurIPS 2021
- MiniMol (pretraining approach): https://arxiv.org/abs/2404.14986
