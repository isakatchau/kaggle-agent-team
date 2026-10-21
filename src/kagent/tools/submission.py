"""Writes a submission that mirrors the competition's sample_submission layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(path: Path, sample: pd.DataFrame, test_ids: pd.Series, pred: np.ndarray,
                     fmt: str, classes: np.ndarray | None, id_column: str) -> Path:
    sub = pd.DataFrame({id_column: test_ids.values})
    target_cols = [c for c in sample.columns if c != id_column]

    if fmt == "proba":
        sub[target_cols[0]] = np.asarray(pred, dtype=float)
    elif fmt == "label":
        if classes is None:  # regression: the prediction is the value
            labels = pred
        else:
            idx = pred.argmax(axis=1) if pred.ndim == 2 else (pred >= 0.5).astype(int)
            labels = classes[idx]
        sub[target_cols[0]] = labels
        sample_dtype = sample[target_cols[0]].dtype
        if pd.api.types.is_integer_dtype(sample_dtype) and classes is not None:
            sub[target_cols[0]] = sub[target_cols[0]].astype(sample_dtype)
    elif fmt == "multi_proba":
        lookup = {str(c): i for i, c in enumerate(classes)}
        for col in target_cols:
            key = next((k for k in (col, col.split("_", 1)[-1]) if k in lookup), None)
            sub[col] = pred[:, lookup[key]] if key is not None else 0.0
    elif fmt == "top3":
        top = np.argsort(-pred, axis=1)[:, :3]
        sub[target_cols[0]] = [" ".join(str(classes[i]) for i in row) for row in top]
    else:
        raise ValueError(f"unknown submission format {fmt}")

    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def check_submission(path: Path, sample: pd.DataFrame) -> list[str]:
    """Cheap pre-flight checks before spending a daily submission."""
    sub = pd.read_csv(path)
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, sample has {len(sample)}")
    if sub.isna().any().any():
        problems.append("submission contains NaN")
    return problems
