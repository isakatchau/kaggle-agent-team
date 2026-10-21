"""Competition metrics on encoded targets.

Predictions are always in "raw" form:
- binary: probability of the positive class, shape (n,)
- multiclass: class probabilities, shape (n, k)
- regression: values in the original target scale, shape (n,)
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

GREATER_IS_BETTER = {
    "accuracy": True,
    "f1": True,
    "roc_auc": True,
    "map@3": True,
    "r2": True,
    "log_loss": False,
    "rmse": False,
    "rmsle": False,
    "mae": False,
}

CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc", "log_loss", "map@3"}
REGRESSION_METRICS = {"rmse", "rmsle", "mae", "r2"}


def _labels(pred: np.ndarray) -> np.ndarray:
    if pred.ndim == 2:
        return pred.argmax(axis=1)
    return (pred >= 0.5).astype(int)


def score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    if metric == "accuracy":
        return float(skm.accuracy_score(y, _labels(pred)))
    if metric == "f1":
        avg = "binary" if pred.ndim == 1 else "macro"
        return float(skm.f1_score(y, _labels(pred), average=avg))
    if metric == "roc_auc":
        if pred.ndim == 2:
            return float(skm.roc_auc_score(y, pred, multi_class="ovr"))
        return float(skm.roc_auc_score(y, pred))
    if metric == "log_loss":
        p = np.clip(pred, 1e-15, 1 - 1e-15)
        labels = np.arange(p.shape[1]) if p.ndim == 2 else [0, 1]
        return float(skm.log_loss(y, p, labels=labels))
    if metric == "map@3":
        top3 = np.argsort(-pred, axis=1)[:, :3]
        hits = top3 == y[:, None]
        weights = np.array([1.0, 1 / 2, 1 / 3])
        return float((hits * weights).sum(axis=1).mean())
    if metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y, pred)))
    if metric == "rmsle":
        return float(np.sqrt(skm.mean_squared_log_error(y, np.clip(pred, 0, None))))
    if metric == "mae":
        return float(skm.mean_absolute_error(y, pred))
    if metric == "r2":
        return float(skm.r2_score(y, pred))
    raise ValueError(f"unsupported metric {metric}")


def is_better(metric: str, a: float | None, b: float | None, min_delta: float = 0.0) -> bool:
    """True when score `a` beats `b` by at least `min_delta`."""
    if a is None:
        return False
    if b is None:
        return True
    if GREATER_IS_BETTER[metric]:
        return a > b + min_delta
    return a < b - min_delta
