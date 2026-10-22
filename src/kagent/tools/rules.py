"""Typed prediction rules, the form many public tutorials take (e.g. Titanic's "women survive,
men die, except ..."). An agent writes a rule as data (`RuleModel`); this module evaluates it.
No generated code is executed.

A rule predicts the majority class of each value of one `base_column` on the training rows, then
applies group overrides: rows matching `when` are set to `set_to` if every other labelled member of
their group (rows sharing `group_keys` and matching `members_when`) had outcome `if_all`. All label
statistics come from the training rows passed in, so the rule can be cross-validated fold by fold.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..agents.schemas import Condition, RuleModel
from . import metrics as M


def _frame(rule: RuleModel, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    full = pd.concat([train, test], ignore_index=True)
    for r in rule.derive:
        full[r.name] = full[r.column].astype(str).str.extract(r.pattern, expand=False).str.strip()
    return full


def _mask(full: pd.DataFrame, conds: list[Condition]) -> np.ndarray:
    m = np.ones(len(full), dtype=bool)
    for c in conds:
        col = full[c.column]
        v = c.value
        if c.op == "isna":
            m &= col.isna().to_numpy()
        elif c.op == "notna":
            m &= col.notna().to_numpy()
        elif c.op in ("in", "not_in"):
            hit = col.astype(str).isin([str(x) for x in (v if isinstance(v, list) else [v])]).to_numpy()
            m &= hit if c.op == "in" else ~hit
        elif c.op in ("==", "!="):
            hit = (col.astype(str) == str(v)) if isinstance(v, str) else (pd.to_numeric(col, errors="coerce") == v)
            m &= hit.to_numpy() if c.op == "==" else ~hit.to_numpy()
        else:
            num = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                m &= {"<": num < v, "<=": num <= v, ">": num > v, ">=": num >= v}[c.op]
    return m


def validate_rule(rule: RuleModel, columns: list[str]) -> list[str]:
    """Problems that make a rule unusable: unknown columns or non-binary outcomes."""
    known = set(columns) | {r.name for r in rule.derive}
    issues = [f"derive: unknown column {r.column}" for r in rule.derive if r.column not in columns]
    cols = [rule.base_column] + [c.column for o in rule.overrides for c in o.when]
    cols += [k for o in rule.overrides for k in o.group_keys]
    cols += [c.column for o in rule.overrides for alt in o.members_when for c in alt]
    issues += [f"unknown column {c}" for c in sorted(set(cols) - known)]
    issues += [f"override outcomes must be 0/1: {o.if_all}->{o.set_to}" for o in rule.overrides
               if o.if_all not in (0, 1) or o.set_to not in (0, 1)]
    return issues


def predict_rule(rule: RuleModel, full: pd.DataFrame, y_full: np.ndarray, known: np.ndarray,
                 rows: np.ndarray) -> np.ndarray:
    """Predict `rows` of `full` using labels of the `known` rows only (y_full elsewhere is ignored)."""
    base = full[rule.base_column].astype(str)
    majority = pd.Series(y_full[known]).groupby(base.iloc[known].to_numpy()).mean()
    prior = float(y_full[known].mean())
    pred = (base.iloc[rows].map(majority).fillna(prior).to_numpy() >= 0.5).astype(int)
    is_known = np.zeros(len(full), dtype=bool)
    is_known[known] = True
    for o in rule.overrides:
        members = np.zeros(len(full), dtype=bool)
        for alt in (o.members_when or [[]]):
            members |= _mask(full, alt)
        parts = full[o.group_keys].astype("string")
        key = parts.fillna("").agg("|".join, axis=1).where(parts.notna().all(axis=1) & members)
        size = key.map(key.value_counts())
        key = key.where(size >= o.min_group_size)
        lab = pd.DataFrame({"key": key[is_known], "y": y_full[is_known]}).dropna(subset=["key"])
        stats = lab.groupby("key")["y"].agg(["sum", "count"])
        k = key.iloc[rows]
        n = k.map(stats["count"]).fillna(0).to_numpy()
        s = k.map(stats["sum"]).fillna(0).to_numpy()
        own = is_known[rows] & k.notna().to_numpy() & members[rows]  # a known row is its own group-mate
        n, s = n - own, s - np.where(own, y_full[rows], 0)
        unanimous = (n >= 1) & (s == (n if o.if_all == 1 else 0))
        hit = unanimous & _mask(full.iloc[rows], o.when)
        pred = np.where(hit, o.set_to, pred)
    return pred


def run_rule_cv(rule: RuleModel, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, metric: str,
                name: str, folds: int, seed: int, repeats: int = 1):
    """Cross-validate a rule like any model; returns an ExperimentResult (model='rule')."""
    from .train import ExperimentResult

    t0 = time.time()
    full = _frame(rule, train, test)
    n = len(train)
    y_full = np.concatenate([y, np.zeros(len(test), dtype=y.dtype)])
    oof, fold_scores = np.zeros(n), []
    for r in range(repeats):
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(np.zeros(n), y):
            p = predict_rule(rule, full, y_full, tr, va)
            oof[va] += p / repeats
            fold_scores.append(M.score(metric, y[va], p.astype(float)))
    test_pred = predict_rule(rule, full, y_full, np.arange(n), np.arange(n, len(full))).astype(float)
    return ExperimentResult(
        name=name, model="rule", params=rule.model_dump(), cv_score=M.score(metric, y, oof),
        fold_scores=fold_scores, oof=oof, test_pred=test_pred, seconds=time.time() - t0, metric=metric,
        repeats=repeats, complexity=float(1 + 2 * len(rule.overrides)),
    )
