"""Run configuration.

Switching competitions means pointing at a different YAML file. Only the Kaggle
slug is required; everything else (target, id column, task, metric) is inferred
by the Planner from the data and the competition's evaluation page, and can be
overridden here when the inference is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Metric = Literal[
    "accuracy", "f1", "roc_auc", "log_loss", "map@3", "rmse", "rmsle", "mae", "r2"
]


class LLMConfig(BaseModel):
    provider: Literal["openai", "mock"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2


class Budget(BaseModel):
    max_iterations: int = Field(3, ge=1, le=10)
    max_submissions: int = Field(3, ge=1, le=10)
    max_experiments_per_iteration: int = Field(4, ge=1, le=8)
    min_improvement: float = Field(0.0005, ge=0)


class RunConfig(BaseModel):
    competition: str
    # Optional overrides. Leave empty to let the agents infer them.
    metric: Metric | None = None
    target: str | None = None
    id_column: str | None = None
    target_percentile: float = Field(0.20, gt=0, le=1)
    cv_folds: int = Field(5, ge=2, le=10)
    research_notebooks: int = Field(
        8, ge=0, le=12, description="Public notebooks the Research agent may read (0 disables research)")
    research_attempts: int = Field(
        3, ge=1, le=5, description="Independent translations of the notebooks into rules; all valid ones compete")
    cv_repeats: int | None = Field(
        None, ge=1, le=5,
        description="Repeated CV with new shuffles. None = 3 below 5,000 train rows (where one split is "
                    "noisier than the differences between candidates), else 1.")
    seed: int = 42
    sample_rows: int | None = Field(
        None, description="Train on a random subset to keep runs fast on big datasets."
    )
    keep_checkpoints: Literal["best", "all"] = Field(
        "best", description="'best' keeps only the fold models behind the best candidate")
    llm: LLMConfig = LLMConfig()
    budget: Budget = Budget()
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**raw)
