import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdmolops


# Bond type vocabulary
BOND_TYPES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
NO_BOND_IDX = 4  # For non-bonded atom pairs


def compute_bond_type_matrix(mol):
    """
    Compute bond type matrix for a molecule.

    Args:
        mol: rdkit.Chem.Mol object

    Returns:
        torch.LongTensor of shape [n_atoms, n_atoms] with bond type indices
    """
    n_atoms = mol.GetNumAtoms()
    bond_type_matrix = torch.full((n_atoms, n_atoms), NO_BOND_IDX, dtype=torch.long)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_idx = BOND_TYPES.get(bond_type, NO_BOND_IDX)
        bond_type_matrix[i, j] = bond_idx
        bond_type_matrix[j, i] = bond_idx

    return bond_type_matrix


def compute_distance_matrix(mol, max_distance=6):
    """
    Compute shortest-path distance matrix, clamped to max_distance.

    Args:
        mol: rdkit.Chem.Mol object
        max_distance: Maximum distance to encode (larger distances clamped)

    Returns:
        torch.LongTensor of shape [n_atoms, n_atoms] with clamped distances
    """
    dist_matrix = torch.tensor(rdmolops.GetDistanceMatrix(mol), dtype=torch.long)
    dist_matrix = torch.clamp(dist_matrix, 0, max_distance)
    return dist_matrix


class EdgeBiasEncoder(nn.Module):
    """
    Computes edge-biased attention weights based on bond types and graph distances.
    Similar to Graphormer's spatial encoding.
    """

    def __init__(self, n_heads=4, max_distance=6, num_bond_types=5):
        """
        Args:
            n_heads: Number of attention heads (one bias per head)
            max_distance: Maximum shortest-path distance to encode
            num_bond_types: Number of bond types (single, double, triple, aromatic, none)
        """
        super().__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        self.num_bond_types = num_bond_types

        # Learnable embeddings for bond types and distances
        # Output dimension is n_heads (one scalar bias per attention head)
        self.bond_type_bias = nn.Embedding(num_bond_types, n_heads)
        self.distance_bias = nn.Embedding(max_distance + 1, n_heads)

    def forward(self, mol):
        """
        Compute edge bias matrix for a molecule.

        Args:
            mol: rdkit.Chem.Mol object

        Returns:
            torch.Tensor of shape [n_atoms, n_atoms, n_heads] with bias values
        """
        device = self.bond_type_bias.weight.device

        # Get bond type and distance matrices
        bond_types = compute_bond_type_matrix(mol).to(device)  # [n, n]
        distances = compute_distance_matrix(mol, self.max_distance).to(device)  # [n, n]

        # Embed and sum biases
        bond_bias = self.bond_type_bias(bond_types)  # [n, n, n_heads]
        dist_bias = self.distance_bias(distances)     # [n, n, n_heads]

        edge_bias = bond_bias + dist_bias  # [n, n, n_heads]

        return edge_bias

    def forward_batch(self, mols, max_atoms=None):
        """
        Compute edge bias for a batch of molecules with padding.

        Args:
            mols: List[rdkit.Chem.Mol] molecules
            max_atoms: Maximum number of atoms (for padding). If None, uses max in batch.

        Returns:
            torch.Tensor of shape [batch, max_atoms, max_atoms, n_heads]
            torch.Tensor of shape [batch, max_atoms] padding mask (True for valid atoms)
        """
        device = self.bond_type_bias.weight.device
        batch_size = len(mols)

        if max_atoms is None:
            max_atoms = max(mol.GetNumAtoms() for mol in mols)

        # Initialize with zeros (will be masked anyway)
        edge_biases = torch.zeros(batch_size, max_atoms, max_atoms, self.n_heads, device=device)
        padding_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)

        for i, mol in enumerate(mols):
            n_atoms = mol.GetNumAtoms()
            bias = self.forward(mol)  # [n_atoms, n_atoms, n_heads]
            edge_biases[i, :n_atoms, :n_atoms, :] = bias
            padding_mask[i, :n_atoms] = True

        return edge_biases, padding_mask
