# What

This is meant to be an interpretable spin on the succesful MPNN based GNN architecture in cheminformatics. Models such as chemprop or MiniMol have proven to be very succesful, probably because graphs are natural (if incomplete) representations of molecules. A major downside is that these models are generally poorly interpretable which is particularly key when used in structure-activity relationship (SAR) landscapes, where interpretable predictions could guide a chemists decision as to what molecules to test next.

## Interpretability Approach: Integrated Gradients

Integrated gradients (Sundararajan et al., ICML, 2017) is a method for attributing a deep neural network's prediction to its input features. The core idea is to integrate the gradients of the output taken along a linear path from a baseline input to the input at hand. Mathematically, for a neural network F(x), an input x and baseline input x' (e.g. the zero input), the attribution for the ith feature is:

IntegratedGrads_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x')) / ∂x_i) dα

The method satisfies important axioms like sensitivity (if inputs differ in one feature but have different predictions, that feature should receive attribution) and implementation invariance (attributions are identical for functionally equivalent networks).

### Reference Baseline for SAR

We extend the standard zero baseline with **MCS-guided reference baselines**:
- Compare an analog to a hit structure using MCS (Maximum Common Substructure) alignment
- Matched atoms: attributed against their hit counterpart → shows context change
- New atoms: attributed against zero → shows full impact
- This directly answers "what makes this analog different from the hit?"

# Why

The long-term (ideal version) goal of this model would be a pre-trained, transformer-on-graph architecture that can be fine-tuned on biological activity data (namely activity as a drug, toxicity, etc.). Using an integrated gradients approach to interpretability (see above) we can see which functional groups (or atoms, scaffolds, etc) are helpful or detrimental to the drugs performance. Based on this we could more efficiently exploit the SAR landscape. The ideal interface would be a Chemdraw style editor where a chemist could come up with new structures and immediately see whether the score(s) improved or not and which atoms positively/negatively contribute to it.

# How

* Keep the codebase as simple as possible. Use as many standard, well-maintained, common packages (transformers, pytorch, scipy, etc) as possible.
* Always keep in mind how these models get deployed in the end: They will pre-train on very large, supervised datasets (including hyperparameter optimization) which have multiple labels. Multi-task learning would be ideal where we train end-to-end including the prediction heads (a separate one for each task), these get deleted after pre-training and we only keep the shared parameters lower down. In general, we will use the same datasets as the pre-trained GNN MiniMol and wherever unclear we will make the same architectural decisions (https://github.com/graphcore-research/minimol, https://arxiv.org/pdf/2404.14986).
* Then the models get fine-tuned on some (presumably smaller) dataset and is then tasked with outputting a binary prediction (between 0 and 1) as to how active a given molecule will be.
* Ultimately the model should be light-weight, fast, and easy to interface.
* Assume that all molecules will be inputted as SMILES. For now, we don't need to support other formats. Do check whether a SMILES is valid and we will have to do graph construction as well.
* Make sure there is way to preconstruct all graphs of a dataset beforehand. Otherwise, each time we will have to reconstruct the same graphs from the SMILES of the molecules even though we could have reused this. Essentially like a pre-tokenized dataset.

# Pretraining Data

**Source file:** `datasets/all_datasets_fused_standardized.parquet`

This is the master pretraining dataset containing:
- ~1.5M molecules
- ~3,288 tasks (assay endpoints)
- SMILES column: `SMILES_std` (standardized)
- Wide format with NaN for missing labels

Use this dataset for all pretraining runs. The multi-task data loader handles missing labels via masked loss.

# Current Implementation Status

## Completed

- [x] Edge-biased attention (Graphormer-style) with bond type and distance encoding
- [x] Atom featurization with bond context
- [x] Multi-task data loader with masked loss for missing labels
- [x] Support for CSV and Parquet data files
- [x] Integrated Gradients interpretability with MCS-based reference baseline
- [x] `interpret_comparison(hit, analog)` for SAR analysis
- [x] Hyperparameter optimization with Ray Tune
- [x] Model save/load
- [x] Unit tests

## TODO

- [ ] Pretrain on `all_datasets_fused_standardized.parquet`
- [ ] Graph caching / pre-tokenization for faster data loading
- [ ] Fine-tuning workflow (load pretrained weights, swap head)
- [ ] Learning rate scheduler (cosine annealing)
- [ ] Gradient accumulation for large effective batch sizes
- [ ] Mixed precision training (fp16/bf16)
- [ ] Distributed training support

# Infrastructure Constraints

- For this project, never allocate more than **16 total EC2 vCPUs**.
- Do not max out the account/project `G and VT` on-demand vCPU quota (32), because other projects run in parallel.

## Active Host Lock (2026-03-08)

- Use **only this EC2 instance** for current InterpreMol setup and runs:
  - Instance ID: `i-0ed3ae87b8a3d1a20`
  - Public IP: `3.220.174.83`
  - Type: `g5.4xlarge` (1 GPU, 16 vCPU)
- Do **not** SSH into or modify any other instance for this project until this lock is explicitly changed.
