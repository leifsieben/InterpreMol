from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader_name: str
    task_type: str
    primary_metric: str


DATASETS: Dict[str, DatasetSpec] = {
    "bbbp": DatasetSpec("bbbp", "load_bbbp", "classification", "roc_auc"),
    "bace": DatasetSpec("bace", "load_bace_classification", "classification", "roc_auc"),
    "clintox": DatasetSpec("clintox", "load_clintox", "classification", "roc_auc"),
    "hiv": DatasetSpec("hiv", "load_hiv", "classification", "roc_auc"),
    "muv": DatasetSpec("muv", "load_muv", "classification", "roc_auc"),
    "pcba": DatasetSpec("pcba", "load_pcba", "classification", "roc_auc"),
    "sider": DatasetSpec("sider", "load_sider", "classification", "roc_auc"),
    "tox21": DatasetSpec("tox21", "load_tox21", "classification", "roc_auc"),
    "toxcast": DatasetSpec("toxcast", "load_toxcast", "classification", "roc_auc"),
    "esol": DatasetSpec("esol", "load_delaney", "regression", "rmse"),
    "freesolv": DatasetSpec("freesolv", "load_freesolv", "regression", "rmse"),
    "lipo": DatasetSpec("lipo", "load_lipo", "regression", "rmse"),
    "qm7": DatasetSpec("qm7", "load_qm7", "regression", "rmse"),
    "qm8": DatasetSpec("qm8", "load_qm8", "regression", "rmse"),
    "qm9": DatasetSpec("qm9", "load_qm9", "regression", "rmse"),
}


def get_spec(dataset_name: str) -> DatasetSpec:
    key = dataset_name.lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {sorted(DATASETS)}")
    return DATASETS[key]


def _get_loader(spec: DatasetSpec):
    return getattr(dc.molnet, spec.loader_name)


def _load_raw_dataset(spec: DatasetSpec, data_dir: Path):
    loader = _get_loader(spec)
    tasks, datasets, _transformers = loader(
        featurizer="Raw",
        splitter=None,
        transformers=[],
        reload=True,
        data_dir=str(data_dir),
        save_dir=str(data_dir / "deepchem_cache"),
    )
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one unsplit dataset for {spec.name}, got {len(datasets)}")
    return tasks, datasets[0]


def _split_dataset(dataset, split: str, seed: int):
    split = split.lower()
    if split == "random":
        splitter = dc.splits.RandomSplitter()
    elif split == "scaffold":
        splitter = dc.splits.ScaffoldSplitter()
    else:
        raise ValueError(f"Unsupported split '{split}'. Use random or scaffold.")

    return splitter.train_valid_test_split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=seed,
    )


def _dataset_to_frame(dataset, tasks: List[str]) -> pd.DataFrame:
    smiles = list(dataset.ids)
    y = np.asarray(dataset.y)
    w = np.asarray(dataset.w)
    if y.ndim == 1:
        y = y[:, None]
    if w.ndim == 1:
        w = w[:, None]

    rows = []
    for smi, targets, weights in zip(smiles, y, w):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = {"smiles": smi}
        for task_name, value, weight in zip(tasks, targets, weights):
            row[task_name] = float(value) if float(weight) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_moleculenet_split(
    dataset_name: str,
    split: str,
    seed: int,
    data_dir: str | Path,
) -> Tuple[DatasetSpec, List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec(dataset_name)
    tasks, dataset = _load_raw_dataset(spec, data_dir)
    train, val, test = _split_dataset(dataset, split=split, seed=seed)
    return (
        spec,
        tasks,
        _dataset_to_frame(train, tasks),
        _dataset_to_frame(val, tasks),
        _dataset_to_frame(test, tasks),
    )


def save_split_csvs(
    output_dir: str | Path,
    tasks: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{split_name}.csv"
        frame[["smiles", *tasks]].to_csv(path, index=False)
        paths[split_name] = str(path)
    return paths


def save_combined_split_csv(
    output_path: str | Path,
    tasks: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for split_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part = frame[["smiles", *tasks]].copy()
        part["split"] = split_name
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(output_path, index=False)
    return str(output_path)
