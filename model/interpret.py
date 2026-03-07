import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw
from rdkit.Chem.Draw import SimilarityMaps
from captum.attr import IntegratedGradients
from IPython.display import Image

from model import GraphTransformerEncoder, GraphEncoder, MLPHead
from atom_embedding import AtomFeaturizer


def load_model(config):
    """Initialize model components from config (for standalone use)."""
    featurizer = AtomFeaturizer(d_model=config["d_model"])

    encoder_model = GraphTransformerEncoder(
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        dim_ff=config["dim_ff"],
        dropout=config["dropout"]
    )

    encoder = GraphEncoder(
        featurizer, encoder_model,
        use_cls_token=config.get("use_cls_token", True),
        use_edge_bias=config.get("use_edge_bias", True),
        max_distance=config.get("max_distance", 6)
    )
    head = MLPHead(
        input_dim=config["d_model"],
        hidden_dim=config["mlp_hidden_dim"],
        depth=config["mlp_head_depth"]
    )

    return encoder.eval(), head.eval()


class ForwardWrapper(torch.nn.Module):
    """Wrapper for Captum IntegratedGradients."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, atom_emb):
        encoded = self.encoder.encode_from_emb(atom_emb)
        pooled = encoded[:, 0] if self.encoder.use_cls_token else encoded.mean(dim=1)
        return self.head(pooled)


def align_molecules_mcs(reference_mol, query_mol):
    """
    Use Maximum Common Substructure to find atom correspondence.

    Args:
        reference_mol: RDKit Mol object (hit/reference)
        query_mol: RDKit Mol object (analog/query)

    Returns:
        dict: mapping query_atom_idx -> reference_atom_idx (or None for new atoms)
    """
    # Find MCS
    mcs_result = rdFMCS.FindMCS(
        [reference_mol, query_mol],
        matchValences=True,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10
    )

    if mcs_result.numAtoms == 0:
        # No common substructure found
        return {i: None for i in range(query_mol.GetNumAtoms())}

    # Get the MCS as a molecule
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    # Get atom mappings
    ref_match = reference_mol.GetSubstructMatch(mcs_mol)
    query_match = query_mol.GetSubstructMatch(mcs_mol)

    # Build mapping: query_idx -> reference_idx
    mapping = {i: None for i in range(query_mol.GetNumAtoms())}

    if ref_match and query_match:
        for mcs_idx, (ref_idx, query_idx) in enumerate(zip(ref_match, query_match)):
            mapping[query_idx] = ref_idx

    return mapping


def create_reference_baseline(model, reference_mol, query_mol):
    """
    Create a baseline embedding for the query molecule using the reference as baseline.

    For atoms that match between reference and query (via MCS):
        - Use the reference atom's embedding as baseline
    For new atoms in the query:
        - Use zero as baseline (shows full contribution)

    Args:
        model: InterpreMol model
        reference_mol: RDKit Mol object (hit/reference)
        query_mol: RDKit Mol object (analog/query)

    Returns:
        torch.Tensor: baseline embedding [1, n_query_atoms, d_model]
    """
    device = next(model.parameters()).device

    # Get embeddings
    ref_emb = model.encoder.embed(reference_mol).to(device)  # [1, n_ref, d_model]
    query_n_atoms = query_mol.GetNumAtoms()
    d_model = ref_emb.shape[-1]

    # Get atom alignment
    mapping = align_molecules_mcs(reference_mol, query_mol)

    # Build baseline
    baseline = torch.zeros(1, query_n_atoms, d_model, device=device)

    for query_idx, ref_idx in mapping.items():
        if ref_idx is not None:
            baseline[0, query_idx, :] = ref_emb[0, ref_idx, :]

    return baseline


def calculate_aspect_ratio(molecule, base_size):
    """Calculate canvas dimensions based on molecule aspect ratio."""
    conf = molecule.GetConformer()
    atom_positions = [conf.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
    x_coords = [pos.x for pos in atom_positions]
    y_coords = [pos.y for pos in atom_positions]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    aspect_ratio = width / height if height > 0 else 1

    canvas_width = max(base_size, int(base_size * aspect_ratio)) if aspect_ratio > 1 else base_size
    canvas_height = max(base_size, int(base_size / aspect_ratio)) if aspect_ratio < 1 else base_size

    return canvas_width, canvas_height


def interpret_smiles(smiles, model, target=0, reference_smiles=None, bw=True, padding=0.05, verbose=True):
    """
    Compute Integrated Gradients attributions for a molecule.

    Args:
        smiles: SMILES string of molecule to interpret
        model: Trained InterpreMol model
        target: Output index for multi-task models (default 0)
        reference_smiles: Optional SMILES string to use as baseline instead of zeros.
                         Useful for comparing analogs to a hit structure.
        bw: Use black and white atom palette
        padding: Padding for molecule drawing
        verbose: Print attribution summary

    Returns:
        (Image, np.ndarray): Visualization and per-atom attribution scores
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)

    device = next(model.parameters()).device
    atom_emb = model.encoder.embed(mol).to(device).requires_grad_()

    wrapper = ForwardWrapper(model.encoder, model.head)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)

    # Choose baseline
    if reference_smiles is not None:
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        baseline = create_reference_baseline(model, ref_mol, mol)
    else:
        baseline = torch.zeros_like(atom_emb)

    # Compute attributions
    attributions, delta = ig.attribute(
        atom_emb, baseline, target=target, return_convergence_delta=True
    )

    # Aggregate attributions per atom (signed!)
    scores = attributions.squeeze(0).sum(dim=1).detach().cpu().numpy()

    if verbose:
        print(f"Attributions for SMILES: {smiles}")
        if reference_smiles:
            print(f"  Reference: {reference_smiles}")
        print(f"  Min attribution: {scores.min():.4f}")
        print(f"  Max attribution: {scores.max():.4f}")
        print(f"  Mean attribution: {scores.mean():.4f}")
        print(f"  Convergence delta: {delta.item():.6f}")

    # Visualization
    base_size = 400
    width, height = calculate_aspect_ratio(mol, base_size)
    drawer = Draw.MolDraw2DCairo(width, height)
    drawer.drawOptions().padding = padding
    if bw:
        drawer.drawOptions().useBWAtomPalette()

    SimilarityMaps.GetSimilarityMapFromWeights(mol, scores.tolist(), draw2d=drawer)
    drawer.FinishDrawing()

    return Image(data=drawer.GetDrawingText()), scores


def interpret_comparison(hit_smiles, analog_smiles, model, target=0, bw=True, padding=0.05):
    """
    Compare an analog to a hit structure using the hit as reference baseline.

    This shows what makes the analog different from the hit:
    - New atoms (not in hit): attributed against zero -> shows full impact
    - Matched atoms: attributed against hit counterpart -> shows context change

    Args:
        hit_smiles: SMILES of the hit/reference structure
        analog_smiles: SMILES of the analog to interpret
        model: Trained InterpreMol model
        target: Output index for multi-task models
        bw: Use black and white atom palette
        padding: Padding for molecule drawing

    Returns:
        (Image, np.ndarray): Visualization and per-atom attribution scores
    """
    return interpret_smiles(
        analog_smiles,
        model,
        target=target,
        reference_smiles=hit_smiles,
        bw=bw,
        padding=padding,
        verbose=True
    )
