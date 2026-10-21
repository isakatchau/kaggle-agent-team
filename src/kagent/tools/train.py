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
from .features import FeatureMatrix

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

    def summary(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "iteration": self.iteration,
            "feature_version": self.feature_version,
            "cv_score": round(self.cv_score, 6),
            # Ensembles are scored once on the full OOF, so they have no fold spread.
            "cv_std": None if self.model == "ensemble" else round(float(np.std(self.fold_scores)), 6),
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


def _predict(est, X, task):
    if task == "regression":
        return est.predict(X)
    proba = est.predict_proba(X)
    return proba[:, 1] if task == "binary" else proba


def run_cv(fm: FeatureMatrix, y: np.ndarray, task: str, metric: str, name: str, model: str,
           params: dict, folds: int, seed: int, log_target: bool = False,
           checkpoint_dir: Path | None = None) -> ExperimentResult:
    params, notes = clip_params(model, params)
    t0 = time.time()
    cat_cols = fm.categorical
    num_cols = [c for c in fm.X_train.columns if c not in cat_cols]
    X = _prepare(fm.X_train, model, cat_cols)
    X_test = _prepare(fm.X_test, model, cat_cols)

    n_classes = int(y.max()) + 1 if task == "multiclass" else 0
    shape = (len(y), n_classes) if n_classes else (len(y),)
    oof = np.zeros(shape)
    test_pred = np.zeros((len(X_test), n_classes) if n_classes else (len(X_test),))

    splitter = (StratifiedKFold if task != "regression" else KFold)(folds, shuffle=True, random_state=seed)
    fold_scores, fitted, importances = [], [], []
    y_fit = np.log1p(y) if (log_target and task == "regression") else y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in splitter.split(X, y):
            est = _estimator(model, task, params, cat_cols, num_cols, seed)
            est.fit(X.iloc[tr], y_fit[tr])
            p_va = _predict(est, X.iloc[va], task)
            p_te = _predict(est, X_test, task)
            if log_target and task == "regression":
                p_va, p_te = np.expm1(p_va), np.expm1(p_te)
            oof[va] = p_va
            test_pred += p_te / folds
            fold_scores.append(M.score(metric, y[va], p_va))
            fitted.append(est)
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)

    importance = {}
    if importances:
        imp = np.mean(importances, axis=0)
        imp = imp / (imp.sum() or 1)
        importance = dict(sorted(zip(X.columns, imp.round(4).tolist()), key=lambda kv: -kv[1])[:25])

    ckpt = None
    if checkpoint_dir is not None:
        ckpt = str(checkpoint_dir / f"{name}.joblib")
        joblib.dump({"models": fitted, "columns": list(X.columns), "params": params,
                     "log_target": log_target}, ckpt, compress=3)

    return ExperimentResult(
        name=name, model=model, params=params, cv_score=M.score(metric, y, oof),
        fold_scores=fold_scores, oof=oof, test_pred=test_pred, seconds=time.time() - t0,
        feature_importance=importance, notes=notes, checkpoint=ckpt,
    )


def blend(results: list[ExperimentResult], y: np.ndarray, metric: str, name: str,
          max_steps: int = 20) -> ExperimentResult:
    """Greedy forward ensemble selection with replacement (Caruana et al., 2004).

    Starts from the best single model and repeatedly adds whichever member most
    improves the out-of-fold score of the running average; stops when nothing
    helps. Weights are the selection counts. Selection uses OOF predictions, so
    the reported CV is mildly optimistic; the evaluator sees it as a candidate
    like any other.
    """
    sign = 1 if M.GREATER_IS_BETTER[metric] else -1
    ranked = sorted(results, key=lambda r: -sign * r.cv_score)
    counts = {ranked[0].name: 1}
    total = ranked[0].oof.astype(float).copy()
    best = M.score(metric, y, total)
    by_name = {r.name: r for r in ranked}
    for _ in range(max_steps):
        n = sum(counts.values())
        trial = [(M.score(metric, y, (total + r.oof) / (n + 1)), r) for r in ranked]
        score, pick = max(trial, key=lambda t: sign * t[0])
        if sign * score <= sign * best + 1e-7:
            break
        best, total = score, total + pick.oof
        counts[pick.name] = counts.get(pick.name, 0) + 1

    n = sum(counts.values())
    weights = {k: round(v / n, 3) for k, v in counts.items()}
    oof = sum(by_name[k].oof * w for k, w in weights.items())
    test = sum(by_name[k].test_pred * w for k, w in weights.items())
    return ExperimentResult(
        name=name, model="ensemble", params={"weights": weights},
        cv_score=M.score(metric, y, oof), fold_scores=[M.score(metric, y, oof)],
        oof=oof, test_pred=test, seconds=0.0,
    )
