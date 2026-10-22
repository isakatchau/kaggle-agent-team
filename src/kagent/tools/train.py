"""Cross-validated training.

The Modeling agent proposes (model, params). Params are clipped to the bounds
below before training, so an LLM cannot request a 50k-tree model or an unknown
argument.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from . import metrics as M
from .features import FeatureMatrix, TargetGroups, group_target_encode

# name -> (type, low, high, default)
PARAM_SPACE: dict[str, dict[str, tuple]] = {
    "lightgbm": {
        "n_estimators": (int, 50, 3000, 600),
        "learning_rate": (float, 0.005, 0.3, 0.03),
        "num_leaves": (int, 4, 256, 31),
        "max_depth": (int, -1, 16, -1),
        "min_child_samples": (int, 1, 200, 20),
        "subsample": (float, 0.4, 1.0, 0.8),
        "colsample_bytree": (float, 0.2, 1.0, 0.8),
        "reg_alpha": (float, 0.0, 10.0, 0.0),
        "reg_lambda": (float, 0.0, 10.0, 1.0),
    },
    "hist_gb": {
        "max_iter": (int, 50, 2000, 400),
        "learning_rate": (float, 0.01, 0.3, 0.05),
        "max_leaf_nodes": (int, 4, 128, 31),
        "min_samples_leaf": (int, 1, 200, 20),
        "l2_regularization": (float, 0.0, 10.0, 0.0),
    },
    "random_forest": {
        "n_estimators": (int, 50, 800, 300),
        "max_depth": (int, 2, 40, 12),
        "min_samples_leaf": (int, 1, 50, 2),
        "max_features": (float, 0.1, 1.0, 0.5),
    },
    "extra_trees": {
        "n_estimators": (int, 50, 800, 300),
        "max_depth": (int, 2, 40, 14),
        "min_samples_leaf": (int, 1, 50, 2),
        "max_features": (float, 0.1, 1.0, 0.6),
    },
    "linear": {
        "C": (float, 0.001, 100.0, 1.0),
        "alpha": (float, 0.001, 1000.0, 1.0),
    },
}


def clip_params(model: str, params: dict) -> tuple[dict, list[str]]:
    space = PARAM_SPACE[model]
    out, notes = {}, []
    for key, (typ, lo, hi, default) in space.items():
        value = params.get(key, default)
        try:
            value = typ(value)
        except (TypeError, ValueError):
            notes.append(f"{key}={value!r} invalid, using {default}")
            value = default
        clipped = min(max(value, lo), hi)
        if clipped != value:
            notes.append(f"{key} clipped {value} -> {clipped}")
        out[key] = clipped
    for key in set(params) - set(space):
        notes.append(f"ignored unknown param {key}")
    return out, notes


@dataclass
class ExperimentResult:
    name: str
    model: str
    params: dict
    cv_score: float
    fold_scores: list[float]
    oof: np.ndarray = field(repr=False)
    test_pred: np.ndarray = field(repr=False)
    seconds: float
    iteration: int = 0
    feature_version: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    metric: str = ""

    @property
    def cv_se(self) -> float:
        # folds overlap across repeats, so this understates the true error; it is a ranking penalty
        return float(np.std(self.fold_scores) / np.sqrt(max(len(self.fold_scores), 1)))

    @property
    def cv_lcb(self) -> float:
        sign = 1 if M.GREATER_IS_BETTER.get(self.metric, True) else -1
        return self.cv_score - sign * self.cv_se

    def summary(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "iteration": self.iteration,
            "feature_version": self.feature_version,
            "cv_score": round(self.cv_score, 6),
            "cv_std": round(float(np.std(self.fold_scores)), 6),
            "cv_se": round(self.cv_se, 6),
            # pessimistic bound, one standard error towards "worse": what the submit gate compares
            "cv_lcb": round(self.cv_lcb, 6),
            "seconds": round(self.seconds, 1),
            "params": self.params,
            "notes": self.notes,
        }


def _estimator(model: str, task: str, p: dict, cat_cols: list[str], num_cols: list[str], seed: int):
    clf = task in ("binary", "multiclass")
    if model == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        return cls(**p, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1)
    if model == "hist_gb":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(**p, categorical_features="from_dtype", random_state=seed)

    # The remaining models need numeric input.
    if model == "linear":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                max_categories=30)), cat_cols),
        ])
        est = LogisticRegression(C=p["C"], max_iter=2000) if clf else Ridge(alpha=p["alpha"])
        return make_pipeline(pre, est)

    trees = {
        ("random_forest", True): RandomForestClassifier,
        ("random_forest", False): RandomForestRegressor,
        ("extra_trees", True): ExtraTreesClassifier,
        ("extra_trees", False): ExtraTreesRegressor,
    }[(model, clf)]
    return trees(**p, random_state=seed, n_jobs=-1)


def _prepare(X: pd.DataFrame, model: str, cat_cols: list[str]) -> pd.DataFrame:
    if model in ("random_forest", "extra_trees"):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].cat.codes
        return X.fillna(-999)
    if model == "linear":
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
        return X
    if model == "hist_gb":
        # HistGradientBoosting rejects categoricals with more than 255 levels.
        X = X.copy()
        for c in cat_cols:
            if len(X[c].cat.categories) > 250:
                X[c] = X[c].cat.codes
        return X
    return X


def _add_group_encoding(groups: list[TargetGroups], Y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                        frames: list[pd.DataFrame], model: str) -> list[pd.DataFrame]:
    """Append this fold's group target encodings to its (train, validation, test) frames."""
    encoded = [group_target_encode(g, Y, tr, va) for g in groups]
    cols = [c for g in groups for c in g.columns(Y.shape[1])]
    out = []
    for i, frame in enumerate(frames):
        enc = pd.DataFrame(np.hstack([e[i] for e in encoded]), columns=cols, index=frame.index)
        out.append(pd.concat([frame, enc.fillna(-999) if model in ("random_forest", "extra_trees") else enc],
                             axis=1))
    return out


def _predict(est, X, task):
    if task == "regression":
        return est.predict(X)
    proba = est.predict_proba(X)
    return proba[:, 1] if task == "binary" else proba


def run_cv(fm: FeatureMatrix, y: np.ndarray, task: str, metric: str, name: str, model: str,
           params: dict, folds: int, seed: int, log_target: bool = False,
           checkpoint_dir: Path | None = None, repeats: int = 1) -> ExperimentResult:
    """K-fold CV, optionally repeated with new shuffles.

    With repeats, every row gets one OOF prediction per repeat and they are averaged, as are
    the test predictions of all folds x repeats models. fold_scores has folds x repeats entries.
    """
    params, notes = clip_params(model, params)
    t0 = time.time()
    cat_cols = fm.categorical
    X = _prepare(fm.X_train, model, cat_cols)
    X_test = _prepare(fm.X_test, model, cat_cols)

    n_classes = int(y.max()) + 1 if task == "multiclass" else 0
    shape = (len(y), n_classes) if n_classes else (len(y),)
    oof = np.zeros(shape)
    test_pred = np.zeros((len(X_test), n_classes) if n_classes else (len(X_test),))

    split_cls = StratifiedKFold if task != "regression" else KFold
    fold_scores, fitted, importances = [], [], []
    y_fit = np.log1p(y) if (log_target and task == "regression") else y

    # Group target encodings use labels, so they are built per fold from that fold's training rows only.
    groups = fm.target_groups
    Y = np.eye(n_classes)[y] if n_classes else np.asarray(y_fit, dtype=float).reshape(-1, 1)
    columns = list(X.columns) + [c for g in groups for c in g.columns(Y.shape[1])]
    num_cols = [c for c in columns if c not in cat_cols]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(repeats):
            for tr, va in split_cls(folds, shuffle=True, random_state=seed + r).split(X, y):
                X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test
                if groups:
                    X_tr, X_va, X_te = _add_group_encoding(groups, Y, tr, va, [X_tr, X_va, X_te], model)
                est = _estimator(model, task, params, cat_cols, num_cols, seed + r)
                est.fit(X_tr, y_fit[tr])
                p_va = _predict(est, X_va, task)
                p_te = _predict(est, X_te, task)
                if log_target and task == "regression":
                    p_va, p_te = np.expm1(p_va), np.expm1(p_te)
                oof[va] += p_va / repeats
                test_pred += p_te / (folds * repeats)
                fold_scores.append(M.score(metric, y[va], p_va))
                fitted.append(est)
                if hasattr(est, "feature_importances_"):
                    importances.append(est.feature_importances_)

    importance = {}
    if importances:
        imp = np.mean(importances, axis=0)
        imp = imp / (imp.sum() or 1)
        importance = dict(sorted(zip(columns, imp.round(4).tolist()), key=lambda kv: -kv[1])[:25])

    ckpt = None
    if checkpoint_dir is not None:
        ckpt = str(checkpoint_dir / f"{name}.joblib")
        joblib.dump({"models": fitted, "columns": columns, "params": params,
                     "log_target": log_target}, ckpt, compress=3)

    return ExperimentResult(
        name=name, model=model, params=params, cv_score=M.score(metric, y, oof),
        fold_scores=fold_scores, oof=oof, test_pred=test_pred, seconds=time.time() - t0,
        feature_importance=importance, notes=notes, checkpoint=ckpt, metric=metric,
    )


def _greedy(oofs: list[np.ndarray], y: np.ndarray, metric: str, max_steps: int) -> dict[int, float]:
    """Greedy forward selection with replacement on these rows; returns member index -> weight."""
    sign = 1 if M.GREATER_IS_BETTER[metric] else -1
    first = max(range(len(oofs)), key=lambda i: sign * M.score(metric, y, oofs[i]))
    counts = {first: 1}
    total = oofs[first].astype(float).copy()
    best = M.score(metric, y, total)
    for _ in range(max_steps):
        n = sum(counts.values())
        score, pick = max(((M.score(metric, y, (total + o) / (n + 1)), i) for i, o in enumerate(oofs)),
                          key=lambda t: sign * t[0])
        if sign * score <= sign * best + 1e-7:
            break
        best, total = score, total + oofs[pick]
        counts[pick] = counts.get(pick, 0) + 1
    n = sum(counts.values())
    return {i: c / n for i, c in counts.items()}


def blend(results: list[ExperimentResult], y: np.ndarray, metric: str, name: str,
          max_steps: int = 20, folds: int = 5, seed: int = 0) -> ExperimentResult:
    """Greedy forward ensemble selection with replacement (Caruana et al., 2004).

    Starts from the best member and repeatedly adds whichever one most improves the
    out-of-fold score of the running average; stops when nothing helps. Weights are the
    selection counts, chosen on all OOF rows.

    Scoring the ensemble on the rows its weights were chosen on overstates it, most on
    small data with a discrete metric (Titanic accuracy: 0.846 in-sample vs 0.840 nested).
    So the reported CV is nested: weights are re-selected on k-1 folds of the OOF
    rows and scored on the held-out fold. The in-sample number is kept in params.
    """
    oofs = [r.oof for r in results]

    def mix(weights: dict[int, float], preds: list[np.ndarray]) -> np.ndarray:
        return sum(preds[i] * w for i, w in weights.items())

    nested = np.zeros_like(oofs[0], dtype=float)
    fold_scores = []
    for tr, va in KFold(folds, shuffle=True, random_state=seed).split(oofs[0]):
        w = _greedy([o[tr] for o in oofs], y[tr], metric, max_steps)
        nested[va] = mix(w, oofs)[va]
        fold_scores.append(M.score(metric, y[va], nested[va]))

    weights = _greedy(oofs, y, metric, max_steps)
    oof = mix(weights, oofs)
    named = {results[i].name: round(w, 3) for i, w in weights.items()}
    return ExperimentResult(
        name=name, model="ensemble",
        params={"weights": named, "in_sample_cv": round(M.score(metric, y, oof), 6)},
        cv_score=M.score(metric, y, nested), fold_scores=fold_scores,
        oof=oof, test_pred=mix(weights, [r.test_pred for r in results]), seconds=0.0, metric=metric,
    )
