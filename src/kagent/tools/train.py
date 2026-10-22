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
from sklearn.tree import DecisionTreeClassifier

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
    # A shallow tree, overridden only where a row's group is unanimous (see group_override).
    "group_override": {
        "depth": (int, 1, 3, 1),
        "min_known": (int, 1, 5, 1),
        "member_purity": (float, 0.5, 1.0, 1.0),
        "member_min_rate": (float, 0.0, 1.0, 0.0),
        "cell_depth": (int, 1, 3, 2),
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
    repeats: int = 1
    complexity: float = 0.0

    @property
    def cv_se(self) -> float:
        # folds overlap across repeats, so this understates the true error; it is a ranking penalty
        return float(np.std(self.fold_scores) / np.sqrt(max(len(self.fold_scores), 1)))

    @property
    def cv_lcb(self) -> float:
        sign = 1 if M.GREATER_IS_BETTER.get(self.metric, True) else -1
        return self.cv_score - sign * self.cv_se

    @property
    def cv_se1(self) -> float:
        """Textbook standard error for the one-standard-error rule: fold std / sqrt(folds per repeat)."""
        per_repeat = max(len(self.fold_scores) // max(self.repeats, 1), 1)
        return float(np.std(self.fold_scores) / np.sqrt(per_repeat))

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
            "cv_se1": round(self.cv_se1, 6),
            # rough count of fitted decisions; the Evaluator prefers the simplest of statistically tied candidates
            "complexity": round(self.complexity, 1),
            "seconds": round(self.seconds, 1),
            "params": self.params if self.model != "rule" else {
                k: self.params.get(k) for k in ("name", "source", "reported_public_score", "base_column")
            } | {"overrides": len(self.params.get("overrides", []))},
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
    if model == "group_override":
        return group_override(fm, y, task, metric, name, params, folds, seed, repeats)
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
        repeats=repeats, complexity=model_complexity(model, params, len(columns)),
    )


SIGNIFICANCE = 0.05


def paired_p_better(a: ExperimentResult, b: ExperimentResult, y: np.ndarray, metric: str) -> float | None:
    """One-sided exact McNemar test that `a` is more accurate than `b` on the same rows, from their
    out-of-fold predictions (Dietterich, 1998): of the rows exactly one of them gets right, does `a`
    win more than half? Only for accuracy, the one metric that is a sum over rows; None otherwise."""
    if metric != "accuracy":
        return None
    from scipy.stats import binomtest

    def right(r: ExperimentResult) -> np.ndarray:
        pred = r.oof.argmax(axis=1) if r.oof.ndim == 2 else (r.oof >= 0.5).astype(int)
        return pred == y

    ra, rb = right(a), right(b)
    wins, losses = int((ra & ~rb).sum()), int((~ra & rb).sum())
    return 1.0 if wins + losses == 0 else float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)


def model_complexity(model: str, params: dict, n_features: int) -> float:
    """Rough number of fitted decisions, comparable across families only in order of magnitude."""
    p = params
    if model == "lightgbm":
        leaves = p["num_leaves"] if p["max_depth"] <= 0 else min(p["num_leaves"], 2 ** p["max_depth"])
        return float(leaves * p["n_estimators"])
    if model == "hist_gb":
        return float(p["max_leaf_nodes"] * p["max_iter"])
    if model in ("random_forest", "extra_trees"):
        return float(p["n_estimators"] * 2 ** min(p["max_depth"], 12))
    if model == "linear":
        return float(n_features)
    return 0.0


def _wilson_lower(correct: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    phat = correct / n
    return (phat + z * z / (2 * n) - z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _unanimous(groups: list[TargetGroups], y: np.ndarray, src: np.ndarray, dst_codes: list[np.ndarray],
               min_known: int, own: np.ndarray | None = None, own_in: np.ndarray | None = None) -> np.ndarray:
    """Per destination row: the outcome shared by all (>= min_known) labelled *other* members of its
    groups, or -1. Only y[src] is read; with `own`, destination rows are training rows and each row's
    own label is taken back out where `own_in` says the row itself is among the sources. Groups that
    disagree with each other give -1."""
    out = np.full(len(dst_codes[0]), -2)  # -2 = no evidence yet
    for tg, dst in zip(groups, dst_codes):
        codes, ok = tg.train[src], tg.train[src] >= 0
        n_g = max(tg.n_groups, 1)
        cnt = np.bincount(codes[ok], minlength=n_g).astype(float)
        pos = np.bincount(codes[ok], weights=y[src][ok].astype(float), minlength=n_g)
        has = dst >= 0
        idx = np.where(has, dst, 0)
        n, s = np.where(has, cnt[idx], 0.0), np.where(has, pos[idx], 0.0)
        if own is not None:
            mine = has & (own_in if own_in is not None else True)
            n, s = n - mine, s - own * mine
        verdict = np.where((n >= min_known) & (s == 0), 0, np.where((n >= min_known) & (s == n), 1, -2))
        out = np.where(out == -2, verdict, np.where((verdict != -2) & (verdict != out), -1, out))
    return np.where(out == -2, -1, out)


def _order_categories(X: np.ndarray, X_other: list[np.ndarray], cat_idx: list[int], y: np.ndarray,
                      min_count: int) -> list[np.ndarray]:
    """Replace category codes by their positive rate on these (training) rows, the ordering under
    which one threshold gives the best binary split of a categorical for a binary target (CART,
    Breiman et al. 1984). Arbitrary codes would hide e.g. one title among others. Categories unseen
    in training, or with fewer than min_count training rows, get the overall rate: a rare level's rate
    is mostly its own rows' labels, which would let the tree read them back."""
    out = [X.copy()] + [o.copy() for o in X_other]
    prior = float(y.mean())
    for j in cat_idx:
        stats = pd.Series(y).groupby(X[:, j]).agg(["mean", "size"])
        rates = stats["mean"].where(stats["size"] >= min_count)
        for arr in out:
            arr[:, j] = pd.Series(arr[:, j]).map(rates).fillna(prior).to_numpy()
    return out


def group_override(fm: FeatureMatrix, y: np.ndarray, task: str, metric: str, name: str, params: dict,
                   folds: int, seed: int, repeats: int = 1) -> ExperimentResult:
    """A shallow tree whose prediction is replaced only where a row's group is unanimous.

    Binary tasks with a label metric. Groups come from the plan's group_target_encoding keys. In each
    fold: fit a depth-1..3 tree on the training rows; for every (leaf, outcome) cell, the override "set
    rows of this leaf whose other group members all had this outcome to it" is accepted only if, on the
    training rows (each judged by its group-mates without its own label), it is right more often than
    wrong with 95% confidence (Wilson lower bound > 0.5). Validation and test rows are judged by their
    training-fold group-mates. With member_purity < 1, only group-mates in tree leaves at most that pure
    count as evidence: a member whose outcome the tree already predicts confidently says little about
    the group (the public rule counts women and children, not men). member_min_rate does the same by
    outcome: only group-mates in cells whose positive rate is at least that count. Overrides are decided per leaf of a
    second tree of depth cell_depth (>= depth), so a finer cell can be overridden while the base stays simple. This is the shape of the strongest leak-free public Titanic models (a
    gender rule plus a few unanimous-family overrides), expressed without any column name.
    """
    params, notes = clip_params("group_override", params)
    t0 = time.time()
    if task != "binary" or not fm.target_groups:
        raise ValueError("group_override needs a binary task and group_target_encoding keys")
    X = _prepare(fm.X_train, "random_forest", fm.categorical).to_numpy(dtype=float)
    X_test = _prepare(fm.X_test, "random_forest", fm.categorical).to_numpy(dtype=float)
    cat_idx = [i for i, c in enumerate(fm.X_train.columns) if c in fm.categorical]
    groups = fm.target_groups
    oof, test_pred, fold_scores, n_rules = np.zeros(len(y)), np.zeros(len(X_test)), [], []
    for r in range(repeats):
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            leaf_min = max(5, len(tr) // 100)
            X_tr, X_va, X_te = _order_categories(X[tr], [X[va], X_test], cat_idx, y[tr], leaf_min)  # training labels only
            tree = DecisionTreeClassifier(max_depth=params["depth"], min_samples_leaf=leaf_min,
                                          random_state=seed + r).fit(X_tr, y[tr])
            base_tr, base_leaf = tree.predict(X_tr), tree.apply(X_tr)
            rate = {leaf: y[tr][base_leaf == leaf].mean() for leaf in np.unique(base_leaf)}
            cells = tree if params["cell_depth"] <= params["depth"] else DecisionTreeClassifier(
                max_depth=params["cell_depth"], min_samples_leaf=max(5, len(tr) // 100),
                random_state=seed + r).fit(X_tr, y[tr])
            leaf_tr = cells.apply(X_tr)
            cell_rate = {leaf: y[tr][leaf_tr == leaf].mean() for leaf in np.unique(leaf_tr)}
            member = np.array([max(rate[b], 1 - rate[b]) <= params["member_purity"]
                               and cell_rate[c] >= params["member_min_rate"] for b, c in zip(base_leaf, leaf_tr)])
            src = tr[member]
            g_tr = _unanimous(groups, y, src, [g.train[tr] for g in groups], params["min_known"],
                              own=y[tr], own_in=member)
            accepted = set()
            for leaf in np.unique(leaf_tr):
                for c in (0, 1):
                    cell = (leaf_tr == leaf) & (g_tr == c) & (base_tr != c)
                    if cell.sum() >= 3 and _wilson_lower(int((y[tr][cell] == c).sum()), int(cell.sum())) > 0.5:
                        accepted.add((int(leaf), c))
            n_rules.append(len(accepted))

            def predict(rows_X, codes):
                base, leaf = tree.predict(rows_X), cells.apply(rows_X)
                g = _unanimous(groups, y, src, codes, params["min_known"])
                for leaf_id, c in accepted:
                    base = np.where((leaf == leaf_id) & (g == c), c, base)
                return base.astype(float)

            p_va = predict(X_va, [g.train[va] for g in groups])
            oof[va] += p_va / repeats
            test_pred += predict(X_te, [g.test for g in groups]) / (folds * repeats)
            fold_scores.append(M.score(metric, y[va], p_va))
    return ExperimentResult(
        name=name, model="group_override",
        params={**params, "mean_accepted_rules": round(float(np.mean(n_rules)), 2)},
        cv_score=M.score(metric, y, oof), fold_scores=fold_scores, oof=oof, test_pred=test_pred,
        seconds=time.time() - t0, notes=notes, metric=metric, repeats=repeats,
        complexity=float(2 ** params["depth"] + np.mean(n_rules)),  # cell tree only routes overrides
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
          max_steps: int = 20, folds: int = 5, seed: int = 0, top_n: int | None = None) -> ExperimentResult:
    """Greedy forward ensemble selection with replacement (Caruana et al., 2004).

    Starts from the best member and repeatedly adds whichever one most improves the
    out-of-fold score of the running average; stops when nothing helps. Weights are the
    selection counts, chosen on all OOF rows.

    Scoring the ensemble on the rows its weights were chosen on overstates it, most on
    small data with a discrete metric (Titanic accuracy: 0.846 in-sample vs 0.840 nested).
    So the reported CV is nested: weights are re-selected on k-1 folds of the OOF
    rows and scored on the held-out fold. The in-sample number is kept in params. With top_n, the
    pool (the top_n members by score) is also chosen inside each fold: choosing it on all rows let the
    held-out fold pick the pool and inflated the nested CV by about half a point on Titanic.
    """
    oofs = [r.oof for r in results]
    sign = 1 if M.GREATER_IS_BETTER[metric] else -1

    def pool(rows: np.ndarray) -> list[int]:
        order = sorted(range(len(oofs)), key=lambda i: -sign * M.score(metric, y[rows], oofs[i][rows]))
        return order[: top_n or len(oofs)]

    def greedy_on(rows: np.ndarray) -> dict[int, float]:
        members = pool(rows)
        w = _greedy([oofs[i][rows] for i in members], y[rows], metric, max_steps)
        return {members[i]: wt for i, wt in w.items()}

    def mix(weights: dict[int, float], preds: list[np.ndarray]) -> np.ndarray:
        return sum(preds[i] * w for i, w in weights.items())

    nested = np.zeros_like(oofs[0], dtype=float)
    fold_scores = []
    for tr, va in KFold(folds, shuffle=True, random_state=seed).split(oofs[0]):
        w = greedy_on(tr)
        nested[va] = mix(w, oofs)[va]
        fold_scores.append(M.score(metric, y[va], nested[va]))

    weights = greedy_on(np.arange(len(y)))
    in_sample = mix(weights, oofs)
    named = {results[i].name: round(w, 3) for i, w in weights.items()}
    return ExperimentResult(
        name=name, model="ensemble",
        params={"weights": named, "in_sample_cv": round(M.score(metric, y, in_sample), 6)},
        cv_score=M.score(metric, y, nested), fold_scores=fold_scores,
        oof=nested, test_pred=mix(weights, [r.test_pred for r in results]), seconds=0.0, metric=metric,
        complexity=float(sum(results[i].complexity for i in weights)),
    )
