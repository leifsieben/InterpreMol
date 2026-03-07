"""
Minimal test of InterpreMol architecture.
Tests: model forward pass, edge bias, interpretability, multi-task output.

Note: Real training will use large multi-label datasets (same as MiniMol).
This is just a sanity check on synthetic data.
"""

import torch
import numpy as np
from rdkit import Chem

# Local imports
from edge_bias import EdgeBiasEncoder, compute_bond_type_matrix, compute_distance_matrix
from atom_embedding import AtomFeaturizer
from model import InterpreMol, GraphEncoder, GraphTransformerEncoder, MLPHead


def test_edge_bias():
    """Test EdgeBiasEncoder on simple molecules."""
    print("\n=== Testing EdgeBiasEncoder ===")

    encoder = EdgeBiasEncoder(n_heads=4, max_distance=6)

    # Test single molecule
    mol = Chem.MolFromSmiles("CCO")  # ethanol: 3 atoms
    bias = encoder(mol)
    assert bias.shape == (3, 3, 4), f"Expected (3,3,4), got {bias.shape}"
    print(f"  Ethanol edge bias shape: {bias.shape}")

    # Check bond type matrix
    bond_matrix = compute_bond_type_matrix(mol)
    print(f"  Bond type matrix:\n{bond_matrix}")
    # C-C and C-O are single bonds (index 0)
    assert bond_matrix[0, 1] == 0, "C-C should be single bond"
    assert bond_matrix[1, 2] == 0, "C-O should be single bond"

    # Test aromatic molecule
    benzene = Chem.MolFromSmiles("c1ccccc1")
    bond_matrix_bz = compute_bond_type_matrix(benzene)
    assert (bond_matrix_bz[bond_matrix_bz != 4] == 3).all(), "Benzene bonds should be aromatic (3)"
    print(f"  Benzene bonds are aromatic: OK")

    # Test batch
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    biases, mask = encoder.forward_batch(mols)
    print(f"  Batch shape: {biases.shape}, mask shape: {mask.shape}")

    print("  EdgeBiasEncoder: PASSED")


def test_atom_featurizer():
    """Test AtomFeaturizer with bond context."""
    print("\n=== Testing AtomFeaturizer ===")

    featurizer = AtomFeaturizer(d_model=64)

    mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
    emb = featurizer(mol)
    print(f"  Acetic acid embedding shape: {emb.shape}")  # [4, 64]
    assert emb.shape == (4, 64), f"Expected (4, 64), got {emb.shape}"

    # Check that different atoms get different embeddings
    assert not torch.allclose(emb[0], emb[1]), "Different atoms should have different embeddings"

    print("  AtomFeaturizer: PASSED")


def test_model_forward():
    """Test full model forward pass."""
    print("\n=== Testing Model Forward ===")

    config = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_ff": 128,
        "dropout": 0.1,
        "mlp_hidden_dim": 64,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "out_dim": 1,  # single task
    }

    model = InterpreMol.from_config(config)
    model.eval()

    # Single molecule
    mol = Chem.MolFromSmiles("c1ccccc1")
    with torch.no_grad():
        out = model([mol])
    print(f"  Single mol output shape: {out.shape}")  # [1, 1]
    assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    # Batch of molecules
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(=O)O", "CN"]]
    with torch.no_grad():
        out = model(mols)
    print(f"  Batch output shape: {out.shape}")  # [4, 1]
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    print("  Model Forward: PASSED")


def test_multi_task():
    """Test multi-task output (like MiniMol pretraining)."""
    print("\n=== Testing Multi-Task Output ===")

    # Multi-task config (e.g., 10 different property predictions)
    config = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_ff": 128,
        "dropout": 0.1,
        "mlp_hidden_dim": 64,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "out_dim": 10,  # 10 tasks
    }

    model = InterpreMol.from_config(config)
    model.eval()

    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1"]]
    with torch.no_grad():
        out = model(mols)
    print(f"  Multi-task output shape: {out.shape}")  # [2, 10]
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

    print("  Multi-Task: PASSED")


def test_interpretability():
    """Test interpretability with reference baseline."""
    print("\n=== Testing Interpretability ===")

    from interpret import (
        align_molecules_mcs,
        create_reference_baseline,
        ForwardWrapper,
    )

    config = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_ff": 128,
        "dropout": 0.1,
        "mlp_hidden_dim": 64,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "out_dim": 1,
    }

    model = InterpreMol.from_config(config)
    model.eval()

    # Test MCS alignment
    hit = Chem.MolFromSmiles("c1ccccc1")  # benzene
    analog = Chem.MolFromSmiles("c1ccc(F)cc1")  # fluorobenzene

    mapping = align_molecules_mcs(hit, analog)
    print(f"  MCS mapping (analog->hit): {mapping}")

    # 6 carbons should map to benzene, 1 fluorine should map to None
    none_count = sum(1 for v in mapping.values() if v is None)
    print(f"  New atoms in analog (not in hit): {none_count}")
    assert none_count == 1, f"Expected 1 new atom (F), got {none_count}"

    # Test reference baseline creation
    baseline = create_reference_baseline(model, hit, analog)
    print(f"  Reference baseline shape: {baseline.shape}")
    assert baseline.shape[1] == analog.GetNumAtoms(), "Baseline should match analog atom count"

    # Check that the fluorine position has zero baseline
    for idx, ref_idx in mapping.items():
        if ref_idx is None:
            assert torch.allclose(baseline[0, idx], torch.zeros(64)), \
                f"New atom at {idx} should have zero baseline"

    print("  Interpretability: PASSED")


def test_gradient_flow():
    """Test that gradients flow through edge-biased attention."""
    print("\n=== Testing Gradient Flow ===")

    config = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_ff": 128,
        "dropout": 0.0,  # no dropout for gradient test
        "mlp_hidden_dim": 64,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "out_dim": 1,
    }

    model = InterpreMol.from_config(config)
    model.train()

    mols = [Chem.MolFromSmiles("CCO")]
    labels = torch.tensor([[1.0]])

    out = model(mols)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
    loss.backward()

    # Check gradients exist for key parameters
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found!"
    print("  Gradients flow correctly: PASSED")


def test_save_load():
    """Test model save/load."""
    print("\n=== Testing Save/Load ===")
    import tempfile
    import os

    config = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_ff": 128,
        "dropout": 0.1,
        "mlp_hidden_dim": 64,
        "mlp_head_depth": 2,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 6,
        "out_dim": 1,
    }

    model = InterpreMol.from_config(config)
    model.eval()

    mol = Chem.MolFromSmiles("CCO")
    with torch.no_grad():
        out_before = model([mol])

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        path = f.name

    # Load
    loaded = InterpreMol.load(path)
    with torch.no_grad():
        out_after = loaded([mol])

    os.unlink(path)

    assert torch.allclose(out_before, out_after), "Output changed after save/load!"
    print("  Save/Load: PASSED")


def test_multitask_dataloader():
    """Test multi-task data loader with missing labels."""
    print("\n=== Testing Multi-Task DataLoader ===")

    import pandas as pd
    from train import (
        MultiTaskMoleculeDataset,
        multitask_collate,
        masked_bce_loss,
        get_dataloaders_multitask,
    )
    from torch.utils.data import DataLoader

    # Create synthetic multi-task dataset with missing labels
    data = {
        "SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "CN", "CCN"],
        "task1": [1.0, 0.0, 1.0, np.nan, 0.0],  # some missing
        "task2": [0.0, 1.0, np.nan, 1.0, 1.0],  # some missing
        "task3": [1.0, 1.0, 0.0, 0.0, np.nan],  # some missing
    }
    df = pd.DataFrame(data)

    # Test dataset creation
    dataset = MultiTaskMoleculeDataset(df, smiles_col="SMILES")
    assert len(dataset) == 5, f"Expected 5, got {len(dataset)}"
    assert dataset.n_tasks == 3, f"Expected 3 tasks, got {dataset.n_tasks}"
    print(f"  Dataset: {len(dataset)} molecules, {dataset.n_tasks} tasks")

    # Test __getitem__
    mol, labels, mask = dataset[0]
    assert labels.shape == (3,), f"Expected (3,), got {labels.shape}"
    assert mask.shape == (3,), f"Expected (3,), got {mask.shape}"
    print(f"  Item 0: labels={labels.tolist()}, mask={mask.tolist()}")

    # Test collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=multitask_collate)
    mols, labels, masks = next(iter(loader))
    assert len(mols) == 2
    assert labels.shape == (2, 3)
    assert masks.shape == (2, 3)
    print(f"  Batch: {len(mols)} mols, labels shape {labels.shape}")

    # Test masked loss
    preds = torch.randn(2, 3, requires_grad=True)
    loss = masked_bce_loss(preds, labels, masks)
    assert loss.requires_grad, "Loss should require grad"
    assert not torch.isnan(loss), "Loss should not be NaN"
    loss.backward()  # Test gradient computation
    assert preds.grad is not None, "Gradients should flow"
    print(f"  Masked BCE loss: {loss.item():.4f}")

    # Test train/val split
    train_ds, val_ds = MultiTaskMoleculeDataset.train_val_split(
        df, val_frac=0.4, seed=42
    )
    assert len(train_ds) + len(val_ds) == len(df)
    print(f"  Split: {len(train_ds)} train, {len(val_ds)} val")

    print("  Multi-Task DataLoader: PASSED")


def test_multitask_training():
    """Test multi-task training loop (1 epoch on tiny data)."""
    print("\n=== Testing Multi-Task Training ===")

    import pandas as pd
    from train import train_model_multitask

    # Tiny synthetic dataset
    data = {
        "SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "CN", "CCN", "CCC", "CCCC", "c1ccc(O)cc1"],
        "task1": [1.0, 0.0, 1.0, np.nan, 0.0, 1.0, np.nan, 0.0],
        "task2": [0.0, 1.0, np.nan, 1.0, 1.0, np.nan, 0.0, 1.0],
    }
    df = pd.DataFrame(data)

    config = {
        "df": df,
        "smiles_col": "SMILES",
        "val_frac": 0.25,
        "batch_size": 2,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "dim_ff": 64,
        "dropout": 0.0,
        "mlp_hidden_dim": 32,
        "mlp_head_depth": 1,
        "use_cls_token": True,
        "use_edge_bias": True,
        "max_distance": 4,
        "lr": 1e-3,
        "epochs": 2,
        "early_stopping_patience": None,
        "loss": "multitask-bce",
        "device": "cpu",
    }

    model, val_loss, logs = train_model_multitask(config)

    assert len(logs["train_losses"]) == 2, "Should have 2 epochs of losses"
    assert logs["n_tasks"] == 2, "Should have 2 tasks"
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Tasks: {logs['n_tasks']}")

    # Test prediction
    from train import predict_smiles
    preds = predict_smiles(model, ["CCO", "c1ccccc1"], classification=True)
    assert preds.shape == (2, 2), f"Expected (2, 2), got {preds.shape}"
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions should be probabilities"
    print(f"  Predictions shape: {preds.shape}")

    print("  Multi-Task Training: PASSED")


def main():
    print("=" * 50)
    print("InterpreMol Architecture Tests")
    print("=" * 50)

    test_edge_bias()
    test_atom_featurizer()
    test_model_forward()
    test_multi_task()
    test_interpretability()
    test_gradient_flow()
    test_save_load()
    test_multitask_dataloader()
    test_multitask_training()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nNote: Real training uses large multi-label datasets")
    print("(same as MiniMol: ChEMBL, PubChem, etc.)")


if __name__ == "__main__":
    main()
