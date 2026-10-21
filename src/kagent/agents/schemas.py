"""Typed contracts between agents.

Each agent returns one of these models. LLM answers are validated against them,
so a malformed answer is rejected before it reaches a tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Task = Literal["binary", "multiclass", "regression"]
ModelName = Literal["lightgbm", "hist_gb", "random_forest", "extra_trees", "linear"]
SubmissionFormat = Literal["label", "proba", "multi_proba", "top3"]


class CompetitionPlan(BaseModel):
    reasoning: str
    task: Task
    metric: Literal["accuracy", "f1", "roc_auc", "log_loss", "map@3", "rmse", "rmsle", "mae", "r2"]
    target: str
    id_column: str
    submission_format: SubmissionFormat
    candidate_models: list[ModelName] = Field(min_length=1)
    risks: list[str] = []


class RegexFeature(BaseModel):
    column: str
    pattern: str = Field(description="Python regex with one capture group")
    name: str


class FeaturePlan(BaseModel):
    reasoning: str
    drop_columns: list[str] = []
    log_transform_columns: list[str] = []
    datetime_columns: list[str] = []
    frequency_encode_columns: list[str] = []
    text_length_columns: list[str] = []
    regex_features: list[RegexFeature] = []
    ratio_features: list[tuple[str, str]] = []
    group_size_columns: list[list[str]] = Field(
        default=[], description="Each entry: columns whose combined value count becomes a feature"
    )
    add_missing_indicators: bool = True
    log_target: bool = False


class Experiment(BaseModel):
    name: str
    model: ModelName
    params: dict = {}
    reasoning: str = ""


class ExperimentBatch(BaseModel):
    reasoning: str
    experiments: list[Experiment] = Field(min_length=1)
    ensemble_top_k: int = Field(0, ge=0, le=5)


class SubmitDecision(BaseModel):
    reasoning: str
    submit: bool
    candidate: str


class Review(BaseModel):
    reasoning: str
    diagnosis: str
    continue_: bool = Field(alias="continue")
    next_focus: Literal["features", "models", "tuning", "stop"]
    suggestions: list[str] = []

    model_config = {"populate_by_name": True}
