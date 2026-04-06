from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .interpremol_transfer import evaluate_predictions


def morgan_matrix(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    features = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        features[idx] = arr
    return features


def _fit_rf(task_type: str):
    if task_type == "classification":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=0,
        )
    return RandomForestRegressor(
        n_estimators=500,
        max_features=0.5,
        n_jobs=-1,
        random_state=0,
    )


def _fit_xgb(task_type: str):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in the current environment.") from exc

    common = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 8,
        "random_state": 0,
    }
    if task_type == "classification":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            **common,
        )
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        **common,
    )


def run_tree_baseline(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_names: List[str],
    task_type: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x = morgan_matrix(train_df["smiles"].tolist())
    test_x = morgan_matrix(test_df["smiles"].tolist())
    y_true = test_df[task_names].to_numpy(dtype=np.float32)
    y_pred = np.full_like(y_true, np.nan, dtype=np.float32)
    mask = ~np.isnan(y_true)

    for task_idx, task_name in enumerate(task_names):
        train_y = train_df[task_name].to_numpy(dtype=np.float32)
        train_mask = ~np.isnan(train_y)
        if train_mask.sum() == 0:
            continue
        if task_type == "classification":
            unique = np.unique(train_y[train_mask])
            if len(unique) < 2:
                continue
        estimator = _fit_rf(task_type) if model_name == "random_forest" else _fit_xgb(task_type)
        estimator.fit(train_x[train_mask], train_y[train_mask])
        if task_type == "classification":
            y_pred[:, task_idx] = estimator.predict_proba(test_x)[:, 1]
        else:
            y_pred[:, task_idx] = estimator.predict(test_x)

    metrics = evaluate_predictions(task_type, y_true=y_true, y_pred=y_pred, mask=mask)
    prediction_frame = pd.DataFrame({"smiles": test_df["smiles"].tolist()})
    for task_idx, task_name in enumerate(task_names):
        prediction_frame[f"{task_name}__true"] = y_true[:, task_idx]
        prediction_frame[f"{task_name}__pred"] = y_pred[:, task_idx]
    predictions_path = output_dir / f"{model_name}_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)

    payload = {
        "model_name": model_name,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
    }
    with open(output_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
