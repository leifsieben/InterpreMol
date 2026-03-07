"""
Streaming dataset for large parquet files.
Loads data in chunks to avoid OOM issues.
Uses sparse label storage since most labels are NaN.
"""

import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from rdkit import Chem
from typing import Optional, List, Tuple


class StreamingMoleculeDataset(IterableDataset):
    """
    Streams molecules from a parquet file without loading everything into memory.

    Uses PyArrow's row group reading to load data in chunks.
    Labels are stored sparsely (only non-NaN values) to save memory.
    """

    def __init__(
        self,
        parquet_path: str,
        smiles_col: str = "SMILES_std",
        label_cols: Optional[List[str]] = None,
        row_groups: Optional[List[int]] = None,
        shuffle_buffer_size: int = 1000,
        max_tasks: Optional[int] = None,
        task_chunk_size: int = 200,  # Read columns in chunks of this size
    ):
        """
        Args:
            parquet_path: Path to parquet file
            smiles_col: Name of SMILES column
            label_cols: List of label columns (None = auto-detect)
            row_groups: Which row groups to read (for train/val split)
            shuffle_buffer_size: Size of shuffle buffer
            max_tasks: Maximum number of tasks to load (for memory efficiency)
            task_chunk_size: Number of label columns to read at a time
        """
        self.parquet_path = parquet_path
        self.smiles_col = smiles_col
        self.shuffle_buffer_size = shuffle_buffer_size
        self.task_chunk_size = task_chunk_size

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

        print(f"StreamingDataset: {self.total_rows} molecules, {self.n_tasks} tasks, "
              f"{len(self.row_groups)} row groups, chunk_size={task_chunk_size}")

    def _read_row_group_chunked(self, rg_idx: int) -> List[Tuple[str, List[Tuple[int, float]]]]:
        """
        Read a row group with chunked column reading.

        Returns list of (smiles, sparse_labels) where sparse_labels is
        list of (task_idx, value) for non-NaN values only.
        """
        # First read SMILES column
        smiles_table = self.parquet_file.read_row_group(rg_idx, columns=[self.smiles_col])
        smiles_col = smiles_table.column(self.smiles_col)
        n_rows = len(smiles_col)

        # Initialize sparse label storage for each row
        sparse_labels = [[] for _ in range(n_rows)]

        # Read label columns in chunks to avoid memory spike
        for chunk_start in range(0, len(self.label_cols), self.task_chunk_size):
            chunk_end = min(chunk_start + self.task_chunk_size, len(self.label_cols))
            chunk_cols = self.label_cols[chunk_start:chunk_end]

            # Read this chunk of columns
            chunk_table = self.parquet_file.read_row_group(rg_idx, columns=chunk_cols)

            # Extract non-NaN values
            for col_offset, col_name in enumerate(chunk_cols):
                task_idx = chunk_start + col_offset
                col_array = chunk_table.column(col_name)

                # Convert to numpy for efficient NaN checking
                col_np = col_array.to_numpy(zero_copy_only=False)

                # Find non-NaN indices and values
                valid_mask = ~np.isnan(col_np)
                valid_indices = np.where(valid_mask)[0]

                for row_idx in valid_indices:
                    sparse_labels[row_idx].append((task_idx, float(col_np[row_idx])))

            # Free memory
            del chunk_table

        # Combine SMILES with sparse labels
        smiles_list = smiles_col.to_pylist()
        result = [(smiles_list[i], sparse_labels[i]) for i in range(n_rows)]

        return result

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Split row groups among workers
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            row_groups = [rg for i, rg in enumerate(self.row_groups) if i % num_workers == worker_id]
        else:
            row_groups = self.row_groups.copy()

        # Shuffle row groups
        np.random.shuffle(row_groups)

        # Buffer for shuffling
        buffer = []

        for rg_idx in row_groups:
            # Read row group with chunked column reading
            row_data = self._read_row_group_chunked(rg_idx)

            # Process each row
            for smiles, sparse_labels in row_data:
                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Add to buffer (store sparse labels)
                buffer.append((mol, sparse_labels))

                # Yield from buffer when full
                if len(buffer) >= self.shuffle_buffer_size:
                    np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

            # Free memory after processing row group
            del row_data

        # Yield remaining items
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item


def streaming_collate(batch):
    """
    Collate function for streaming dataset.
    Converts sparse labels back to dense tensors.
    """
    mols, sparse_labels_list = zip(*batch)

    batch_size = len(mols)

    # Get n_tasks from the dataset (passed via closure or we infer from max task_idx)
    # For safety, we use a global or infer from data
    if sparse_labels_list and any(sparse_labels_list):
        max_task = max(
            (task_idx for sparse_labels in sparse_labels_list
             for task_idx, _ in sparse_labels),
            default=-1
        )
        n_tasks = max_task + 1
    else:
        n_tasks = 1  # Fallback

    # Convert sparse to dense
    labels = torch.full((batch_size, n_tasks), float('nan'), dtype=torch.float32)

    for i, sparse_labels in enumerate(sparse_labels_list):
        for task_idx, val in sparse_labels:
            labels[i, task_idx] = val

    # Create mask and replace NaN with 0
    masks = ~torch.isnan(labels)
    labels = torch.nan_to_num(labels, nan=0.0)

    return list(mols), labels, masks


def create_streaming_collate_fn(n_tasks: int):
    """Create a collate function with known n_tasks for efficiency."""
    def collate(batch):
        mols, sparse_labels_list = zip(*batch)
        batch_size = len(mols)

        # Pre-allocate with known size
        labels = torch.zeros((batch_size, n_tasks), dtype=torch.float32)
        masks = torch.zeros((batch_size, n_tasks), dtype=torch.bool)

        for i, sparse_labels in enumerate(sparse_labels_list):
            for task_idx, val in sparse_labels:
                labels[i, task_idx] = val
                masks[i, task_idx] = True

        return list(mols), labels, masks

    return collate


def create_streaming_dataloaders(
    parquet_path: str,
    smiles_col: str = "SMILES_std",
    label_cols: Optional[List[str]] = None,
    val_frac: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    max_tasks: Optional[int] = None,
    shuffle_buffer_size: int = 1000,
    task_chunk_size: int = 200,
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
        task_chunk_size=task_chunk_size,
    )

    val_ds = StreamingMoleculeDataset(
        parquet_path, smiles_col, label_cols,
        row_groups=val_rgs,
        shuffle_buffer_size=100,  # Less shuffling for val
        max_tasks=max_tasks,
        task_chunk_size=task_chunk_size,
    )

    # Use collate function with known n_tasks
    collate_fn = create_streaming_collate_fn(train_ds.n_tasks)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
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
        num_workers=0,
        max_tasks=100,  # Test with limited tasks
        task_chunk_size=50,
    )

    print(f"\nTesting train loader (n_tasks={n_tasks})...")
    for i, (mols, labels, masks) in enumerate(train_loader):
        print(f"Batch {i}: {len(mols)} mols, labels {labels.shape}, valid labels: {masks.sum().item()}")
        if i >= 2:
            break

    print("\nStreaming works!")
