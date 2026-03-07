import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict


class AtomFeaturizer(nn.Module):
    ATOM_VOCAB_LIST = [
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'Te', 'At', 'Na', 'K', 'Li', 'Ca', 'Mg',
        'Zn', 'Fe', 'Cu', 'Mn', 'Al', 'As', 'Ba', 'Bi', 'Cd', 'Co',
        'Cr', 'Cs', 'Ga', 'Hg', 'In', 'Ni', 'Pb', 'Rb', 'Sb', 'Sn',
        'Sr', 'Tl', 'V', 'Zr', '*', 'Unknown'
    ]

    # Bond type to index for bond context features
    BOND_TYPE_MAP = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    def __init__(self,
                 degree_vocab=range(6),
                 formal_charge_vocab=range(-5, 6),
                 hybridization_vocab=['SP', 'SP2', 'SP3'],
                 num_h_vocab=range(5),
                 chirality_vocab=[0, 1, 2, 3],  # RDKit chirality values
                 d_model=128):
        super().__init__()

        self.atom_vocab = {sym: i for i, sym in enumerate(self.ATOM_VOCAB_LIST)}

        self.atom_type_embedding = nn.Embedding(len(self.atom_vocab), d_model)
        self.degree_embedding = nn.Embedding(len(degree_vocab), d_model)
        self.charge_embedding = nn.Embedding(len(formal_charge_vocab), d_model)
        self.hybridization_embedding = nn.Embedding(len(hybridization_vocab), d_model)
        self.num_h_embedding = nn.Embedding(len(num_h_vocab), d_model)
        self.chirality_embedding = nn.Embedding(len(chirality_vocab), d_model)

        # Bond context: embedding for each bond type count (0-4 bonds of each type)
        self.bond_context_embedding = nn.Embedding(5, d_model)  # 0-4 bonds per type

        # Scalar features: aromatic, in_ring, mass, + 4 bond type counts
        self.scalar_proj = nn.Linear(7, d_model)

        self.vocab = {
            'atom': self.atom_vocab,
            'hybrid': {k: i for i, k in enumerate(hybridization_vocab)},
            'chirality': {k: i for i, k in enumerate(chirality_vocab)}
        }

    def _get_bond_context(self, atom):
        """Count bonds of each type connected to an atom."""
        bond_counts = [0, 0, 0, 0]  # single, double, triple, aromatic
        for bond in atom.GetBonds():
            bond_type = bond.GetBondType()
            idx = self.BOND_TYPE_MAP.get(bond_type, 0)
            bond_counts[idx] = min(bond_counts[idx] + 1, 4)  # clamp to 4
        return bond_counts

    def forward(self, mol):
        """Returns [n_atoms, d_model] embedding for a molecule"""
        atom_features = []
        device = self.atom_type_embedding.weight.device

        for atom in mol.GetAtoms():
            # Safe fallback for unknown atoms
            atom_symbol = atom.GetSymbol()
            atom_type = self.vocab['atom'].get(atom_symbol, self.vocab['atom']['Unknown'])

            degree = min(atom.GetDegree(), 5)  # clamp to vocab size
            formal_charge = atom.GetFormalCharge() + 5
            formal_charge = max(0, min(formal_charge, 10))  # clamp to vocab size
            hybrid_str = str(atom.GetHybridization())
            hybrid = self.vocab['hybrid'].get(hybrid_str, 0)
            is_aromatic = float(atom.GetIsAromatic())
            in_ring = float(atom.IsInRing())
            num_H = min(atom.GetTotalNumHs(), 4)  # clamp to vocab size
            chirality = self.vocab['chirality'].get(int(atom.GetChiralTag()), 0)
            mass = atom.GetMass() / 100.0

            # Bond context: count of each bond type
            bond_counts = self._get_bond_context(atom)

            # Embed categorical features
            emb = self.atom_type_embedding(torch.tensor(atom_type, dtype=torch.long, device=device)) \
                + self.degree_embedding(torch.tensor(degree, dtype=torch.long, device=device)) \
                + self.charge_embedding(torch.tensor(formal_charge, dtype=torch.long, device=device)) \
                + self.hybridization_embedding(torch.tensor(hybrid, dtype=torch.long, device=device)) \
                + self.num_h_embedding(torch.tensor(num_H, dtype=torch.long, device=device)) \
                + self.chirality_embedding(torch.tensor(chirality, dtype=torch.long, device=device))

            # Add bond context embeddings (sum over bond types)
            for bt_idx, count in enumerate(bond_counts):
                if count > 0:
                    emb = emb + self.bond_context_embedding(torch.tensor(count, dtype=torch.long, device=device))

            # Embed scalar features (including normalized bond counts)
            scalar_feats = torch.tensor(
                [is_aromatic, in_ring, mass] + [c / 4.0 for c in bond_counts],
                dtype=torch.float, device=device
            )
            emb = emb + self.scalar_proj(scalar_feats)

            atom_features.append(emb)

        return torch.stack(atom_features, dim=0)  # [n_atoms, d_model]

def compute_shortest_path_matrix(mol):
    return torch.tensor(rdmolops.GetDistanceMatrix(mol), dtype=torch.long)

def compute_ring_membership(mol):
    ring_info = mol.GetRingInfo()
    ring_membership = set()
    for ring in ring_info.AtomRings():
        ring_membership.update(ring)
    return list(ring_membership)


def count_topological_angles(mol):
    angle_counts = defaultdict(int)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for j in neighbors:
            for k in neighbors:
                if j < k:
                    angle_counts[i] += 1
    return angle_counts
