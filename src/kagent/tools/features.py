"""Applies a FeaturePlan to train and test.

The Data agent only chooses *which* transformations to apply; this module is the
only code that touches the data. Unknown or invalid references in a plan are
skipped and reported, never executed blindly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..agents.schemas import FeaturePlan


# Group target encoding is for small groups of related rows (a ticket, a booking, a household). On a
# category with big groups, "mean of the other members" takes exactly one value per class inside each
# group, so a model that can also see the category reads every training row's label back and learns a
# rule that does not exist at test time (Titanic: ticket prefix and cabin letter, mean group sizes 12
# and 42, dropped LightGBM CV from 0.847 to 0.827 and the public score to 311/418). Categories already
# have their own treatment; keys whose repeated groups are larger than this on average are skipped.
MAX_MEAN_GROUP_SIZE = 10


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))


@dataclass
class TargetGroups:
    """Group ids of one group_target_encoding entry.

    The encoding needs labels, so it is not part of the feature matrix: run_cv computes it inside every
    fold with group_target_encode. build_features only adds the label-free group size.
    """
    name: str
    train: np.ndarray  # group id per train row, -1 = no group
    test: np.ndarray
    smoothing: float = 0.0

    @property
    def n_groups(self) -> int:
        return int(max(self.train.max(initial=-1), self.test.max(initial=-1))) + 1

    def columns(self, n_stats: int) -> list[str]:
        means = [self.name] if n_stats == 1 else [f"{self.name}_{j}" for j in range(n_stats)]
        return means + [f"{self.name}_n"]


@dataclass
class FeatureMatrix:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    categorical: list[str]
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    target_groups: list[TargetGroups] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def validate_plan(plan: FeaturePlan, columns: list[str], numeric: list[str]) -> tuple[FeaturePlan, list[str]]:
    """Drop references to columns that do not exist or have the wrong type."""
    cols, num = set(columns), set(numeric)
    issues: list[str] = []

    def keep(values, allowed, label):
        good = [v for v in values if v in allowed]
        for v in values:
            if v not in allowed:
                issues.append(f"{label}: unknown or invalid column '{v}'")
        return good

    regex = []
    for r in plan.regex_features:
        try:
            ok = r.column in cols and re.compile(r.pattern).groups >= 1
        except re.error:
            ok = False
        (regex.append(r) if ok else issues.append(f"regex_features: invalid '{r.name}'"))

    derived = cols | {r.name for r in regex}
    # every column build_features can create: an encoding's outputs must not collide with any of them,
    # or every model fails later on duplicate feature names while build_features itself succeeds
    produced = {_safe(c) for c in derived | {f"{c}_isna" for c in derived}
                | {f"{c}_len" for c in plan.text_length_columns}
                | {f"{c}_{p}" for c in plan.datetime_columns for p in ("year", "month", "dow")}
                | {f"{c}_freq" for c in plan.frequency_encode_columns}
                | {"size_" + "_".join(g) for g in plan.group_size_columns}
                | {f"{a}_per_{b}" for a, b in plan.ratio_features}}

    def outputs(name: str) -> set[str]:
        return {name, f"{name}_n", f"{name}_size"} | {f"{name}_{j}" for j in range(32)}  # _j: multiclass

    encodings = []
    for g in plan.group_target_encoding:
        name = _safe(g.name)
        if all(k in derived for k in g.keys) and not outputs(name) & produced:
            produced |= outputs(name)
            encodings.append(g.model_copy(update={"name": name}))
        else:
            issues.append(f"group_target_encoding: invalid '{g.name}' (unknown key, or its output columns "
                          "collide with another feature)")

    ratios = [tuple(p) for p in plan.ratio_features if p[0] in num and p[1] in num and p[0] != p[1]]
    issues += [f"ratio_features: invalid pair {p}" for p in plan.ratio_features if tuple(p) not in ratios]
    groups = [g for g in plan.group_size_columns if g and all(c in cols for c in g)]

    clean = plan.model_copy(
        update={
            "drop_columns": keep(plan.drop_columns, derived, "drop_columns"),
            "log_transform_columns": keep(plan.log_transform_columns, num, "log_transform_columns"),
            "datetime_columns": keep(plan.datetime_columns, cols, "datetime_columns"),
            "frequency_encode_columns": keep(plan.frequency_encode_columns, cols, "frequency_encode_columns"),
            "text_length_columns": keep(plan.text_length_columns, cols, "text_length_columns"),
            "regex_features": regex,
            "ratio_features": ratios[:20],
            "group_size_columns": groups[:10],
            "group_target_encoding": encodings[:5],
        }
    )
    return clean, issues


def build_features(train: pd.DataFrame, test: pd.DataFrame, target: str, id_column: str,
                   plan: FeaturePlan) -> FeatureMatrix:
    base_cols = [c for c in train.columns if c not in (target, id_column) and c in test.columns]
    full = pd.concat([train[base_cols], test[base_cols]], axis=0, ignore_index=True)
    n_train = len(train)
    applied: list[str] = []

    for r in plan.regex_features:
        full[r.name] = full[r.column].astype(str).str.extract(r.pattern, expand=False)
        applied.append(f"regex {r.column} -> {r.name}")

    target_groups, skipped = [], []
    for g in plan.group_target_encoding:
        parts = full[g.keys].astype("string")
        key = parts.fillna("").agg("|".join, axis=1).where(parts.notna().all(axis=1))
        codes = pd.Series(pd.factorize(key)[0])  # missing key -> -1
        size = codes.map(codes[codes >= 0].value_counts()).fillna(1)
        repeated = codes[codes >= 0].value_counts()
        mean_size = float(repeated[repeated >= 2].mean()) if (repeated >= 2).any() else 0.0
        if mean_size > MAX_MEAN_GROUP_SIZE:
            skipped.append(f"group target encoding {g.keys}: mean group size {mean_size:.0f} > "
                           f"{MAX_MEAN_GROUP_SIZE}, a category rather than groups of related rows")
            continue
        codes = codes.where(size >= g.min_group_size, -1).to_numpy()
        name = _safe(g.name)
        full[f"{name}_size"] = size.to_numpy(dtype=float)
        target_groups.append(TargetGroups(name, codes[:n_train], codes[n_train:], g.smoothing))
        applied.append(f"group target encoding {g.keys} -> {name}")

    for col in plan.text_length_columns:
        full[f"{col}_len"] = full[col].astype(str).str.len()
        applied.append(f"text length {col}")

    for col in plan.datetime_columns:
        dt = pd.to_datetime(full[col], errors="coerce")
        if dt.notna().mean() > 0.5:
            full[f"{col}_year"] = dt.dt.year
            full[f"{col}_month"] = dt.dt.month
            full[f"{col}_dow"] = dt.dt.dayofweek
            full = full.drop(columns=[col])
            applied.append(f"datetime parts {col}")

    for col in plan.frequency_encode_columns:
        if col in full:
            full[f"{col}_freq"] = full[col].map(full[col].value_counts(dropna=False)).astype(float)
            applied.append(f"frequency encode {col}")

    for group in plan.group_size_columns:
        key = full[group].astype("string").fillna("__nan__").agg("|".join, axis=1)
        full["size_" + "_".join(group)] = key.map(key.value_counts()).astype(float)
        applied.append(f"group size {group}")

    for a, b in plan.ratio_features:
        if a in full and b in full:
            full[f"{a}_per_{b}"] = full[a] / (full[b].replace(0, np.nan))
            applied.append(f"ratio {a}/{b}")

    for col in plan.log_transform_columns:
        if col in full and (full[col].dropna() >= 0).all():
            full[col] = np.log1p(full[col])
            applied.append(f"log1p {col}")

    drop = [c for c in plan.drop_columns if c in full]
    full = full.drop(columns=drop)
    if drop:
        applied.append(f"drop {drop}")

    if plan.add_missing_indicators:
        for col in [c for c in full.columns if full[c].isna().any()]:
            if full[col].isna().mean() > 0.01:
                full[f"{col}_isna"] = full[col].isna().astype(int)

    categorical = []
    for col in full.columns:
        if not pd.api.types.is_numeric_dtype(full[col]):
            full[col] = full[col].astype("string").fillna("__nan__").astype("category")
            categorical.append(col)
        elif pd.api.types.is_bool_dtype(full[col]):
            full[col] = full[col].astype(int)

    full.columns = [_safe(c) for c in full.columns]
    categorical = [_safe(c) for c in categorical]
    return FeatureMatrix(
        X_train=full.iloc[:n_train].reset_index(drop=True),
        X_test=full.iloc[n_train:].reset_index(drop=True),
        categorical=categorical,
        applied=applied,
        skipped=skipped,
        target_groups=target_groups,
    )


def group_target_encode(tg: TargetGroups, Y: np.ndarray, tr: np.ndarray,
                        va: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold-safe encoding of one group spec for one CV fold: (training rows, validation rows, test rows).

    Each row gets the mean of Y over the *other* labelled members of its group, then their count (0 = no
    information, mean NaN). Only Y[tr] is read: validation and test rows use every training-fold row, and a
    training-fold row the other training-fold rows, so its own label never enters its group statistic. That
    is what a test row sees too; splitting the training fold into inner folds instead lost CV on Titanic.
    Y is (n, k): the target, or one-hot classes.
    """
    n_groups, k = max(tg.n_groups, 1), Y.shape[1]
    # One prior per fold. A per-row prior without the row's own label, (sum - y_i) / (N - 1), would take
    # exactly one value per class and let a model read y_i back from any smoothed mean.
    prior = Y[tr].mean(axis=0)

    def encode(src: np.ndarray, dst: np.ndarray, own: np.ndarray | None = None) -> np.ndarray:
        codes, ok = tg.train[src], tg.train[src] >= 0
        count = np.bincount(codes[ok], minlength=n_groups).astype(float)
        sums = np.column_stack([np.bincount(codes[ok], Y[src][ok, j], minlength=n_groups) for j in range(k)])
        has = dst >= 0
        n, s = np.where(has, count[np.where(has, dst, 0)], 0.0), sums[np.where(has, dst, 0)] * has[:, None]
        if own is not None:  # destination rows are among the sources: take each row's own label back out
            n, s = n - has, s - own * has[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (s + tg.smoothing * prior) / (n + tg.smoothing)[:, None]
        mean[n == 0] = np.nan
        return np.column_stack([mean, n])

    return encode(tr, tg.train[tr], own=Y[tr]), encode(tr, tg.train[va]), encode(tr, tg.test)
