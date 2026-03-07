"""
Streaming dataset for large parquet files.
Loads data in chunks to avoid OOM issues.
"""

import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from rdkit import Chem
from typing import Optional, List


class StreamingMoleculeDataset(IterableDataset):
    """
    Streams molecules from a parquet file without loading everything into memory.

    Uses PyArrow's row group reading to load data in chunks.
    """

    def __init__(
        self,
        parquet_path: str,
        smiles_col: str = "SMILES_std",
        label_cols: Optional[List[str]] = None,
        row_groups: Optional[List[int]] = None,
        shuffle_buffer_size: int = 1000,
        max_tasks: Optional[int] = None,
    ):
        """
        Args:
            parquet_path: Path to parquet file
            smiles_col: Name of SMILES column
            label_cols: List of label columns (None = auto-detect)
            row_groups: Which row groups to read (for train/val split)
            shuffle_buffer_size: Size of shuffle buffer
            max_tasks: Maximum number of tasks to load (for memory efficiency)
        """
        self.parquet_path = parquet_path
        self.smiles_col = smiles_col
        self.shuffle_buffer_size = shuffle_buffer_size

        # Open parquet file to get metadata
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.parquet_file.metadata.num_row_groups
        self.total_rows = self.parquet_file.metadata.num_rows

        # Get column names
        schema = self.parquet_file.schema_arrow
        all_columns = [field.name for field in schema]

        # Filter label columns
        if label_cols is None:
            self.label_cols = [c for c in all_columns if 'smiles' not in c.lower()]
        else:
            self.label_cols = label_cols

        # Limit tasks if requested (for memory efficiency)
        if max_tasks is not None and len(self.label_cols) > max_tasks:
            print(f"Limiting tasks from {len(self.label_cols)} to {max_tasks}")
            self.label_cols = self.label_cols[:max_tasks]

        self.n_tasks = len(self.label_cols)

        # Which row groups to read
        if row_groups is None:
            self.row_groups = list(range(self.num_row_groups))
        else:
            self.row_groups = row_groups

        print(f"StreamingDataset: {self.total_rows} molecules, {self.n_tasks} tasks, {len(self.row_groups)} row groups")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Split row groups among workers
        if worker_info is not None:
            # Multi-worker: each worker gets a subset of row groups
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            row_groups = [rg for i, rg in enumerate(self.row_groups) if i % num_workers == worker_id]
        else:
            row_groups = self.row_groups

        # Shuffle row groups
        np.random.shuffle(row_groups)

        # Buffer for shuffling within row groups
        buffer = []

        for rg_idx in row_groups:
            # Read row group
            table = self.parquet_file.read_row_group(
                rg_idx,
                columns=[self.smiles_col] + self.label_cols
            )
            df = table.to_pandas()

            # Process each row
            for idx in range(len(df)):
                smiles = df[self.smiles_col].iloc[idx]
                labels = df[self.label_cols].iloc[idx].values.astype(np.float32)

                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Create mask for non-NaN labels
                mask = ~np.isnan(labels)

                # Add to buffer
                buffer.append((mol, labels, mask))

                # Yield from buffer when full
                if len(buffer) >= self.shuffle_buffer_size:
                    np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        # Yield remaining items
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item


def streaming_collate(batch):
    """Collate function for streaming dataset."""
    mols, labels, masks = zip(*batch)

    labels = torch.tensor(np.stack(labels), dtype=torch.float32)
    masks = torch.tensor(np.stack(masks), dtype=torch.bool)

    return list(mols), labels, masks


def create_streaming_dataloaders(
    parquet_path: str,
    smiles_col: str = "SMILES_std",
    label_cols: Optional[List[str]] = None,
    val_frac: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    max_tasks: Optional[int] = None,
    shuffle_buffer_size: int = 1000,
):
    """
    Create train/val dataloaders that stream from parquet.

    Splits by row groups (not individual rows) for efficiency.
    """
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.metadata.num_row_groups

    # Split row groups into train/val
    all_rgs = list(range(num_row_groups))
    np.random.seed(42)
    np.random.shuffle(all_rgs)

    val_size = max(1, int(num_row_groups * val_frac))
    val_rgs = all_rgs[:val_size]
    train_rgs = all_rgs[val_size:]

    print(f"Train row groups: {len(train_rgs)}, Val row groups: {len(val_rgs)}")

    train_ds = StreamingMoleculeDataset(
        parquet_path, smiles_col, label_cols,
        row_groups=train_rgs,
        shuffle_buffer_size=shuffle_buffer_size,
        max_tasks=max_tasks,
    )

    val_ds = StreamingMoleculeDataset(
        parquet_path, smiles_col, label_cols,
        row_groups=val_rgs,
        shuffle_buffer_size=100,  # Less shuffling for val
        max_tasks=max_tasks,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=streaming_collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=streaming_collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.n_tasks


if __name__ == "__main__":
    # Test streaming
    import sys

    if len(sys.argv) < 2:
        print("Usage: python streaming_dataset.py <parquet_path>")
        sys.exit(1)

    path = sys.argv[1]

    train_loader, val_loader, n_tasks = create_streaming_dataloaders(
        path,
        smiles_col="SMILES_std",
        batch_size=32,
        num_workers=0  # Use 0 for testing
    )

    print(f"\nTesting train loader...")
    for i, (mols, labels, masks) in enumerate(train_loader):
        print(f"Batch {i}: {len(mols)} mols, labels {labels.shape}, valid labels: {masks.sum().item()}")
        if i >= 2:
            break

    print("\nStreaming works!")
