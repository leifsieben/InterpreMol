"""
Graph caching for faster data loading.

Pre-computes molecular graphs and features from SMILES, saving them to disk.
This avoids rebuilding graphs from SMILES every epoch.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm


def smiles_to_hash(smiles: str) -> str:
    """Generate a short hash for a SMILES string."""
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


def process_molecule(smiles: str) -> Optional[Dict]:
    """
    Process a single SMILES string into cacheable features.

    Returns dict with:
        - n_atoms: number of atoms
        - atom_features: dict of feature arrays for AtomFeaturizer
        - bond_type_matrix: [n, n] bond types
        - distance_matrix: [n, n] shortest path distances
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()

    # Bond type vocabulary
    BOND_TYPES = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    NO_BOND = 4

    # Compute bond type matrix
    bond_type_matrix = np.full((n_atoms, n_atoms), NO_BOND, dtype=np.int8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPES.get(bond.GetBondType(), NO_BOND)
        bond_type_matrix[i, j] = bt
        bond_type_matrix[j, i] = bt

    # Compute distance matrix
    distance_matrix = rdmolops.GetDistanceMatrix(mol).astype(np.int8)

    # Extract atom features (raw indices, not embeddings)
    atom_features = {
        'atom_type': [],
        'degree': [],
        'formal_charge': [],
        'hybridization': [],
        'num_h': [],
        'chirality': [],
        'is_aromatic': [],
        'in_ring': [],
        'mass': [],
        'bond_counts': [],  # [single, double, triple, aromatic] per atom
    }

    ATOM_VOCAB = {
        'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9,
        'B': 10, 'Si': 11, 'Se': 12, 'Te': 13, 'At': 14, 'Na': 15, 'K': 16, 'Li': 17,
        'Ca': 18, 'Mg': 19, 'Zn': 20, 'Fe': 21, 'Cu': 22, 'Mn': 23, 'Al': 24, 'As': 25,
        'Ba': 26, 'Bi': 27, 'Cd': 28, 'Co': 29, 'Cr': 30, 'Cs': 31, 'Ga': 32, 'Hg': 33,
        'In': 34, 'Ni': 35, 'Pb': 36, 'Rb': 37, 'Sb': 38, 'Sn': 39, 'Sr': 40, 'Tl': 41,
        'V': 42, 'Zr': 43, '*': 44, 'Unknown': 45
    }

    HYBRID_VOCAB = {'SP': 0, 'SP2': 1, 'SP3': 2}

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_features['atom_type'].append(ATOM_VOCAB.get(symbol, ATOM_VOCAB['Unknown']))
        atom_features['degree'].append(min(atom.GetDegree(), 5))
        atom_features['formal_charge'].append(max(0, min(atom.GetFormalCharge() + 5, 10)))

        hybrid_str = str(atom.GetHybridization())
        atom_features['hybridization'].append(HYBRID_VOCAB.get(hybrid_str, 0))

        atom_features['num_h'].append(min(atom.GetTotalNumHs(), 4))
        atom_features['chirality'].append(min(int(atom.GetChiralTag()), 3))
        atom_features['is_aromatic'].append(float(atom.GetIsAromatic()))
        atom_features['in_ring'].append(float(atom.IsInRing()))
        atom_features['mass'].append(atom.GetMass() / 100.0)

        # Bond counts
        bond_counts = [0, 0, 0, 0]
        for bond in atom.GetBonds():
            bt = BOND_TYPES.get(bond.GetBondType(), 0)
            bond_counts[bt] = min(bond_counts[bt] + 1, 4)
        atom_features['bond_counts'].append(bond_counts)

    # Convert to numpy arrays
    for key in atom_features:
        atom_features[key] = np.array(atom_features[key], dtype=np.float32 if key in ['is_aromatic', 'in_ring', 'mass'] else np.int8)
    atom_features['bond_counts'] = np.array(atom_features['bond_counts'], dtype=np.int8)

    return {
        'n_atoms': n_atoms,
        'atom_features': atom_features,
        'bond_type_matrix': bond_type_matrix,
        'distance_matrix': distance_matrix,
    }


def build_graph_cache(
    smiles_list: List[str],
    cache_dir: str,
    num_workers: int = 8,
    chunk_size: int = 10000
) -> Tuple[Dict[str, str], List[int]]:
    """
    Build graph cache for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        cache_dir: Directory to save cached graphs
        num_workers: Number of parallel workers
        chunk_size: Number of molecules per cache file

    Returns:
        (mapping dict, list of failed indices)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Process in parallel
    print(f"Processing {len(smiles_list)} molecules with {num_workers} workers...")

    results = []
    failed_indices = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_molecule, smi): i for i, smi in enumerate(smiles_list)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing molecules"):
            idx = futures[future]
            try:
                result = future.result()
                if result is None:
                    failed_indices.append(idx)
                else:
                    results.append((idx, result))
            except Exception as e:
                failed_indices.append(idx)

    # Sort by index
    results.sort(key=lambda x: x[0])

    # Save in chunks
    mapping = {}
    for chunk_idx in range(0, len(results), chunk_size):
        chunk = results[chunk_idx:chunk_idx + chunk_size]
        chunk_file = cache_dir / f"chunk_{chunk_idx:08d}.pkl"

        chunk_data = {idx: data for idx, data in chunk}
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk_data, f)

        for idx, _ in chunk:
            mapping[idx] = str(chunk_file)

    # Save mapping
    mapping_file = cache_dir / "mapping.pkl"
    with open(mapping_file, 'wb') as f:
        pickle.dump(mapping, f)

    # Save failed indices
    if failed_indices:
        failed_file = cache_dir / "failed_indices.pkl"
        with open(failed_file, 'wb') as f:
            pickle.dump(failed_indices, f)
        print(f"Warning: {len(failed_indices)} molecules failed to process")

    print(f"Cached {len(results)} molecules in {cache_dir}")
    return mapping, failed_indices


class CachedMoleculeDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-cached molecular graphs.
    Much faster than building graphs from SMILES on-the-fly.
    """

    def __init__(
        self,
        labels: np.ndarray,
        cache_dir: str,
        valid_indices: Optional[List[int]] = None
    ):
        """
        Args:
            labels: [n_samples, n_tasks] label array (can have NaN)
            cache_dir: Directory with cached graphs
            valid_indices: Optional list of valid indices (excluding failed SMILES)
        """
        self.cache_dir = Path(cache_dir)
        self.labels = labels

        # Load mapping
        with open(self.cache_dir / "mapping.pkl", 'rb') as f:
            self.mapping = pickle.load(f)

        # Filter to valid indices
        if valid_indices is not None:
            self.indices = [i for i in valid_indices if i in self.mapping]
        else:
            self.indices = sorted(self.mapping.keys())

        self.n_tasks = labels.shape[1] if len(labels.shape) > 1 else 1

        # Cache for loaded chunks (LRU-like)
        self._chunk_cache = {}
        self._max_cached_chunks = 10

    def __len__(self):
        return len(self.indices)

    def _load_chunk(self, chunk_file: str) -> Dict:
        """Load a chunk file, using cache."""
        if chunk_file not in self._chunk_cache:
            if len(self._chunk_cache) >= self._max_cached_chunks:
                # Remove oldest entry
                oldest = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest]

            with open(chunk_file, 'rb') as f:
                self._chunk_cache[chunk_file] = pickle.load(f)

        return self._chunk_cache[chunk_file]

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        chunk_file = self.mapping[real_idx]
        chunk_data = self._load_chunk(chunk_file)
        mol_data = chunk_data[real_idx]

        # Get labels
        label_row = self.labels[real_idx]
        if len(self.labels.shape) == 1:
            labels = torch.tensor([label_row], dtype=torch.float)
            mask = torch.tensor([not np.isnan(label_row)], dtype=torch.bool)
        else:
            labels = torch.tensor(label_row, dtype=torch.float)
            mask = ~torch.isnan(labels)

        return mol_data, labels, mask


def cached_collate(batch):
    """
    Collate function for CachedMoleculeDataset.

    Returns:
        mol_data_list: List of mol_data dicts
        labels: [batch, n_tasks]
        masks: [batch, n_tasks]
    """
    mol_data, labels, masks = zip(*batch)
    return list(mol_data), torch.stack(labels), torch.stack(masks)


def build_cache_from_parquet(
    parquet_path: str,
    cache_dir: str,
    smiles_col: str = "SMILES_std",
    num_workers: int = 8,
    chunk_size: int = 10000
):
    """
    Build graph cache from a parquet file.

    Args:
        parquet_path: Path to parquet file
        cache_dir: Directory to save cache
        smiles_col: Name of SMILES column
        num_workers: Number of parallel workers
        chunk_size: Molecules per cache file
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    smiles_list = df[smiles_col].tolist()

    print(f"Found {len(smiles_list)} molecules")

    mapping, failed = build_graph_cache(
        smiles_list,
        cache_dir,
        num_workers=num_workers,
        chunk_size=chunk_size
    )

    # Save metadata
    metadata = {
        'parquet_path': parquet_path,
        'smiles_col': smiles_col,
        'n_molecules': len(smiles_list),
        'n_cached': len(mapping),
        'n_failed': len(failed),
        'columns': list(df.columns),
    }
    with open(Path(cache_dir) / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Cache built: {len(mapping)} molecules cached, {len(failed)} failed")
    return mapping, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build graph cache from parquet file")
    parser.add_argument("parquet_path", help="Path to parquet file")
    parser.add_argument("cache_dir", help="Directory to save cache")
    parser.add_argument("--smiles-col", default="SMILES_std", help="SMILES column name")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Molecules per chunk")

    args = parser.parse_args()

    build_cache_from_parquet(
        args.parquet_path,
        args.cache_dir,
        smiles_col=args.smiles_col,
        num_workers=args.workers,
        chunk_size=args.chunk_size
    )
