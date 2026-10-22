"""Typed contracts between agents.

Each agent returns one of these models. LLM answers are validated against them,
so a malformed answer is rejected before it reaches a tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Task = Literal["binary", "multiclass", "regression"]
ModelName = Literal["lightgbm", "hist_gb", "random_forest", "extra_trees", "linear", "group_override"]
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


class GroupTargetEncoding(BaseModel):
    keys: list[str] = Field(
        min_length=1, max_length=6,
        description="Columns whose exact combined value defines a group: raw columns or names of regex_features "
                    "(e.g. a surname extracted from a name, a booking code). Rows with a missing key part, or "
                    "whose regex did not match, belong to no group",
    )
    name: str = Field(description="Prefix of the new columns <name>, <name>_n, <name>_size")
    min_group_size: int = Field(2, ge=2, le=20, description="Groups with fewer rows (train+test) count as no group")
    smoothing: float = Field(
        0.0, ge=0.0, le=50.0,
        description="Pseudo-count pulling the group mean towards the overall training mean; 0 = plain mean "
                    "of the other members",
    )


class FeaturePlan(BaseModel):
    reasoning: str
    drop_columns: list[str] = Field(default=[], description="Raw columns or regex_features names to remove")
    log_transform_columns: list[str] = []
    datetime_columns: list[str] = []
    frequency_encode_columns: list[str] = []
    text_length_columns: list[str] = []
    regex_features: list[RegexFeature] = []
    ratio_features: list[tuple[str, str]] = []
    group_size_columns: list[list[str]] = Field(
        default=[],
        description="Each entry is a list of columns; adds the number of rows (train+test) sharing that exact "
                    "combination of values. It never adds columns together: [['a','b']] counts rows with the same "
                    "(a, b) pair, it is not a+b",
    )
    group_target_encoding: list[GroupTargetEncoding] = Field(
        default=[],
        description="Rows sharing the same key values form a group (people on one ticket or booking, one "
                    "household, one device). Adds <name> = mean target of the OTHER group members whose label is "
                    "known (one column per class for multiclass; empty when none), <name>_n = how many such "
                    "members there are (0 = no information, unlike a mean of 0), <name>_size = group size over "
                    "train+test. Computed inside every CV fold from training rows only, so a row's own label "
                    "never enters its own value. Use it when an identifier-like column, or part of one, repeats "
                    "across rows, is shared between train and test (profile test_seen_pct), and group members "
                    "plausibly share the outcome. Keys must form SMALL groups of related rows (a full ticket or "
                    "booking code, a surname): a category such as a code prefix, a letter or a class is not a "
                    "group, and keys whose repeated groups average more than 10 rows are skipped. Key columns "
                    "stay in the feature set unless listed in drop_columns",
    )
    add_missing_indicators: bool = True
    log_target: bool = False


class Condition(BaseModel):
    column: str = Field(description="Raw column or a name from `derive`")
    op: Literal["==", "!=", "in", "not_in", "<", "<=", ">", ">=", "isna", "notna"]
    value: str | float | int | list[str | float | int] | None = None


class GroupOverride(BaseModel):
    when: list[Condition] = Field(description="All must hold for a row to be overridden")
    group_keys: list[str] = Field(min_length=1, max_length=6,
                                  description="Columns whose exact combined value defines a group")
    members_when: list[list[Condition]] = Field(
        default=[], description="Which rows count as group members: OR of AND-lists; empty = every row")
    min_group_size: int = Field(2, ge=2, le=20, description="Members over train+test for a group to count")
    if_all: int = Field(description="Outcome every other labelled member must have had (0 or 1)")
    set_to: int = Field(description="Prediction for the overridden rows (0 or 1)")


class RuleModel(BaseModel):
    name: str
    derive: list[RegexFeature] = Field(default=[], description="Columns extracted from raw text first")
    base_column: str = Field(description="Predict the training majority class of each value of this column")
    overrides: list[GroupOverride] = []
    source: str = Field("", description="Notebook the rule comes from")
    reported_public_score: float | None = Field(None, description="Public score the source reports")


class Technique(BaseModel):
    name: str
    summary: str = Field(description="What it does, in at most 20 words")
    reported_public_score: float | None = Field(None, description="Public score the notebook reports, if any")
    source: str = Field(description="Notebook ref (owner/slug)")


class ReadingList(BaseModel):
    reasoning: str
    read: list[str] = Field(default=[], max_length=8, description="Notebook refs (owner/slug) to read, from the listing")
    search_terms: list[str] = Field(default=[], max_length=3,
                                    description="Extra searches among this competition's notebooks")


class ResearchNotes(BaseModel):
    # rules first and short prose: free tiers cap output tokens, and the rules are what the team can use
    rules: list[RuleModel] = Field(default=[], max_length=4,
                                   description="Techniques that are rules over columns, written in this schema; "
                                               "they are cross-validated like any model")
    techniques: list[Technique] = Field(default=[], max_length=5)
    reasoning: str = Field(description="At most two sentences")


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
