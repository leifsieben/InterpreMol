"""
Minimal test of InterpreMol architecture.
Tests: model forward pass, edge bias, interpretability, multi-task output.

Note: Real training will use large multi-label datasets (same as MiniMol).
This is just a sanity check on synthetic data.
"""

import torch
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

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nNote: Real training uses large multi-label datasets")
    print("(same as MiniMol: ChEMBL, PubChem, etc.)")


if __name__ == "__main__":
    main()
