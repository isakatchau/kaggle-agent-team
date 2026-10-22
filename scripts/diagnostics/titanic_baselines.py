"""Manual Titanic diagnostics, outside the agent team: three baselines fixed before seeing any score.

    python scripts/diagnostics/titanic_baselines.py --data data/titanic --out runs/diagnostics

Each baseline gets a repeated stratified CV accuracy (3x5) and a submission file. They test why the
agents' public score stays near 0.77 while their CV is 0.83-0.84:
- gender: females survive, males die. No fitted parameter, so its CV-to-public gap is the part of the
  gap that comes from the test set itself.
- simple: a shallow random forest on six classic features. Compared with the agents' 17-27 features,
  it tests whether the gap grows with feature richness.
- wcg: the public "woman-child-group" rule (C. Deotte, Kaggle): the gender model, except boys in
  woman-child groups where every known member survived (predicted to survive) and females in groups
  where every known member died (predicted to die). Group statistics use training labels only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    full = pd.concat([train, test], ignore_index=True)
    full["Title"] = full["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    full["Surname"] = full["Name"].str.extract(r"^([^,]+),", expand=False)
    full["Boy"] = full["Title"].eq("Master")
    full["WomanChild"] = full["Sex"].eq("female") | full["Boy"]
    # Deotte's group key: surname, class, ticket without its last digit, fare, port
    full["GroupId"] = (full["Surname"] + "-" + full["Pclass"].astype(str) + "-" + full["Ticket"].str[:-1]
                       + "-" + full["Fare"].astype(str) + "-" + full["Embarked"].astype(str))
    full.loc[~full["WomanChild"], "GroupId"] = np.nan  # adult males are not in woman-child groups
    counts = full["GroupId"].value_counts()
    full.loc[full["GroupId"].map(counts).fillna(0) < 2, "GroupId"] = np.nan
    return full


def gender(full: pd.DataFrame, known: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return full.loc[rows, "Sex"].eq("female").astype(int).to_numpy()


def wcg(full: pd.DataFrame, known: np.ndarray, rows: np.ndarray) -> np.ndarray:
    labelled = full.loc[known].dropna(subset=["GroupId"])
    rate = labelled.groupby("GroupId")["Survived"].mean()
    part = full.loc[rows]
    r = part["GroupId"].map(rate)  # NaN: no labelled member in this group
    pred = part["Sex"].eq("female").astype(int).to_numpy().copy()  # pandas 3 returns a read-only view
    pred[(part["Boy"] & r.eq(1.0)).to_numpy()] = 1
    pred[(part["Sex"].eq("female") & r.eq(0.0)).to_numpy()] = 0
    return pred


SIMPLE_TITLES = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}


def simple(full: pd.DataFrame, known: np.ndarray, rows: np.ndarray) -> np.ndarray:
    X = pd.DataFrame({
        "female": full["Sex"].eq("female").astype(int),
        "pclass": full["Pclass"],
        "title": full["Title"].map(SIMPLE_TITLES).fillna(4),
        "family": full["SibSp"] + full["Parch"] + 1,
        "fare": np.log1p(full["Fare"].fillna(full.loc[known, "Fare"].median())),
        "age": full["Age"].fillna(full.loc[known].groupby("Title")["Age"].transform("median"))
                          .fillna(full.loc[known, "Age"].median()),
    })
    X["age"] = X["age"].fillna(full.loc[known, "Age"].median())
    m = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=3, random_state=0)
    m.fit(X.loc[known], full.loc[known, "Survived"].astype(int))
    return m.predict(X.loc[rows])


def cv_accuracy(fn, full: pd.DataFrame, n_train: int, repeats: int = 3, folds: int = 5) -> tuple[float, float]:
    y = full.loc[: n_train - 1, "Survived"].astype(int).to_numpy()
    scores = []
    for r in range(repeats):
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=42 + r).split(np.zeros(n_train), y):
            sub = full.copy()
            sub.loc[va, "Survived"] = np.nan  # validation labels are hidden from the baseline
            scores.append(float((fn(sub, tr, va) == y[va]).mean()))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/titanic")
    p.add_argument("--out", default="runs/diagnostics")
    args = p.parse_args()
    train = pd.read_csv(Path(args.data) / "train.csv")
    test = pd.read_csv(Path(args.data) / "test.csv")
    full, n = prepare(train, test), len(train)
    known, rows = np.arange(n), np.arange(n, len(full))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fn in (("gender", gender), ("simple", simple), ("wcg", wcg)):
        cv, std = cv_accuracy(fn, full, n)
        pred = fn(full, known, rows)
        path = out / f"titanic_{name}.csv"
        pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)}).to_csv(path, index=False)
        report[name] = {"cv": round(cv, 4), "cv_std": round(std, 4), "file": str(path),
                        "test_survivors": int(pred.sum())}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
