"""Generates a local practice competition for offline runs and tests.

The data has the usual Kaggle traps (missing values, a high-cardinality text
column with signal hidden in a prefix, skewed numerics, an interaction). The
leaderboard is synthetic: scores of reference models of increasing quality on
the hidden solution, spread with noise. It exists to exercise the loop, not to
claim a real ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M


def _frame(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    income = rng.lognormal(10, 0.8, n)
    age = rng.integers(18, 80, n).astype(float)
    tenure = rng.exponential(4, n)
    visits = rng.poisson(5, n).astype(float)
    region = rng.choice(["north", "south", "east", "west", "center"], n, p=[.3, .25, .2, .15, .1])
    plan = rng.choice(["basic", "plus", "pro"], n, p=[.5, .35, .15])
    title = rng.choice(["Mr", "Mrs", "Dr", "Ms", "Prof"], n, p=[.4, .3, .1, .15, .05])
    code = [f"{t}. {rng.integers(1000, 9999)}-{rng.choice(list('ABCDEFG'))}" for t in title]
    signup = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")

    latent = (
        0.9 * np.log(income) - 9
        + 0.03 * (age - 45)
        - 0.25 * tenure
        + 0.15 * visits * (plan == "pro")
        + np.select([region == "north", region == "west"], [0.6, -0.5], 0)
        + np.select([title == "Dr", title == "Prof"], [0.8, 1.1], 0)
        + 0.3 * (signup.month.isin([11, 12]))
    )
    df = pd.DataFrame({
        "income": income.round(2), "age": age, "tenure_years": tenure.round(2), "visits": visits,
        "region": region, "plan": plan, "customer_code": code, "signup_date": signup.strftime("%Y-%m-%d"),
    })
    for col, frac in (("income", .08), ("age", .05), ("region", .03)):
        df.loc[rng.random(n) < frac, col] = np.nan
    return df, latent


def make_competition(root: Path, task: str = "binary", n_train: int = 4000, n_test: int = 2000,
                     seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    df, latent = _frame(n_train + n_test, rng)
    n = len(df)
    if task == "binary":
        y = (rng.random(n) < 1 / (1 + np.exp(-(latent + rng.normal(0, 0.7, n))))).astype(int)
        metric, target, fmt_value = "roc_auc", "churned", 0.5
    else:
        y = np.round(np.exp(11 + 0.35 * latent + rng.normal(0, 0.25, n)), 2)
        metric, target, fmt_value = "rmsle", "lifetime_value", float(np.median(y))

    df.insert(0, "id", np.arange(1, n + 1))
    df[target] = y
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], target: fmt_value}).to_csv(root / "sample_submission.csv", index=False)
    test[["id", target]].to_csv(root / "_solution.csv", index=False)

    test_latent = latent[n_train:]
    oracle = 1 / (1 + np.exp(-test_latent)) if task == "binary" else np.exp(11 + 0.35 * test_latent)
    board = _reference_leaderboard(test[target].to_numpy(), oracle, metric, rng)
    pd.DataFrame({"score": board}).to_csv(root / "_leaderboard.csv", index=False)
    evaluation = {
        "roc_auc": "Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.",
        "rmsle": "Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed value.",
    }[metric]
    (root / "_meta.json").write_text(json.dumps({
        "competition": f"synthetic-{task}", "metric": metric, "target": target,
        "id_column": "id", "evaluation": evaluation,
    }, indent=2))
    return root


def _reference_leaderboard(y_true, oracle_pred, metric, rng, n_teams: int = 400) -> list[float]:
    """Teams spread between a constant prediction and the noise-free oracle.

    Quality q ~ Beta(5, 1.6): the median team closes ~78% of the gap and the
    top-20% cut sits near 88%, so a baseline model does not reach it.
    """
    const = np.full(len(y_true), 0.5 if metric == "roc_auc" else float(np.median(y_true)))
    floor = M.score(metric, y_true, const) if metric != "roc_auc" else 0.5
    best = M.score(metric, y_true, oracle_pred)
    q = np.clip(rng.beta(5, 1.6, n_teams), 0, 0.99)
    return (floor + (best - floor) * q).round(5).tolist()


def make_holdout_competition(train_csv: Path, root: Path, target: str, id_column: str, metric: str,
                             evaluation: str = "", test_frac: float = 0.33, seed: int = 7,
                             sample_name: str = "sample_submission.csv") -> Path:
    """Turn any labelled CSV into an offline competition by hiding a holdout's labels.

    Useful to rehearse a real competition's schema before spending submissions.
    There is no leaderboard, so percentile is reported as unknown.
    """
    df = pd.read_csv(train_csv)
    test_mask = np.random.default_rng(seed).random(len(df)) < test_frac
    train, test = df[~test_mask], df[test_mask]
    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(root / "test.csv", index=False)
    placeholder = train[target].mode().iloc[0]
    pd.DataFrame({id_column: test[id_column], target: placeholder}).to_csv(root / sample_name, index=False)
    test[[id_column, target]].to_csv(root / "_solution.csv", index=False)
    (root / "_meta.json").write_text(json.dumps({
        "competition": root.name, "metric": metric, "target": target,
        "id_column": id_column, "evaluation": evaluation}, indent=2))
    return root
