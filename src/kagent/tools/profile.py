"""Dataset profiling: the facts every agent reasons over."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_competition_files(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def find(stem: str) -> Path:
        hits = sorted(p for p in data_dir.glob(f"*{stem}*.csv") if not p.name.startswith("_"))
        if not hits:
            raise FileNotFoundError(f"no *{stem}*.csv in {data_dir}")
        return hits[0]

    train = pd.read_csv(find("train"))
    test = pd.read_csv(find("test"))
    # Titanic ships `gender_submission.csv`, most others `sample_submission.csv`.
    sample = pd.read_csv(find("submission"))
    return train, test, sample


def profile(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> dict:
    columns = []
    for col in train.columns:
        s = train[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "in_test": col in test.columns,
            "n_unique": int(s.nunique()),
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "examples": [str(v) for v in s.dropna().unique()[:5]],
        }
        if col in test.columns:  # can a statistic learned per value on train reach the test rows?
            info["test_seen_pct"] = round(float(test[col].isin(s.dropna().unique()).mean()) * 100, 2)
        if pd.api.types.is_numeric_dtype(s):
            info["min"] = float(np.nanmin(s)) if s.notna().any() else None
            info["max"] = float(np.nanmax(s)) if s.notna().any() else None
            info["skew"] = round(float(s.skew()), 2) if s.notna().sum() > 2 else None
        columns.append(info)

    target_candidates = [c for c in train.columns if c not in test.columns]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_columns": train.shape[1],
        "columns": columns,
        "target_candidates": target_candidates,
        "sample_submission": {
            "columns": list(sample.columns),
            "dtypes": {c: str(t) for c, t in sample.dtypes.items()},
            "head": sample.head(3).astype(str).to_dict(orient="records"),
        },
    }


def compact_profile(prof: dict, max_columns: int = 50, max_examples: int = 3) -> dict:
    """Smaller view for LLM prompts on wide datasets (free tiers cap tokens per request).

    Target candidates and sample-submission columns are always kept: a truncated view once dropped
    House Prices' SalePrice, which the Data agent's heuristic then could not find.
    """
    keep = set(prof.get("target_candidates", [])) | set(prof.get("sample_submission", {}).get("columns", []))
    first = [c for c in prof["columns"] if c["name"] in keep]
    rest = [c for c in prof["columns"] if c["name"] not in keep]
    chosen = first + rest[: max(max_columns - len(first), 0)]
    out = dict(prof)
    out["columns"] = [{k: (v[:max_examples] if k == "examples" else v) for k, v in c.items() if v is not None}
                      for c in chosen]
    if len(prof["columns"]) > len(chosen):
        out["columns_truncated"] = len(prof["columns"]) - len(chosen)
    return out
