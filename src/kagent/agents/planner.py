from __future__ import annotations

import re

from ..tools import metrics as M
from .base import Agent
from .schemas import CompetitionPlan


def _col(profile: dict, name: str) -> dict | None:
    return next((c for c in profile["columns"] if c["name"] == name), None)


def metric_from_text(text: str) -> str | None:
    t = text.lower()
    rules = [
        (r"map@3|mean average precision @ ?3", "map@3"),
        (r"(rmse|root.mean.squared.error).{0,120}logarithm|rmsle|squared log", "rmsle"),
        (r"log.?loss|logarithmic loss|cross.entropy", "log_loss"),
        (r"area under the roc|roc.auc|\bauc\b", "roc_auc"),
        (r"\brmse\b|root.mean.squared.error", "rmse"),
        (r"mean absolute error|\bmae\b", "mae"),
        (r"\bf1\b|f-score|f1.score", "f1"),
        (r"\br2\b|r\^2|coefficient of determination", "r2"),
        (r"accuracy|percentage of .* correctly", "accuracy"),
    ]
    for pattern, metric in rules:
        if re.search(pattern, t):
            return metric
    return None


LARGE = 200_000  # rows; above this, forests are too slow and their checkpoints too big


class PlannerAgent(Agent[CompetitionPlan]):
    name = "planner"
    schema = CompetitionPlan
    role = (
        "Read the dataset profile, the sample submission and the competition's evaluation text. "
        "Decide the task type, the exact metric, the target and id columns, the submission format "
        "(label = class labels or regression values, proba = probability of the positive class, "
        "multi_proba = one probability column per class, top3 = space-separated top-3 labels), "
        "which model families are worth trying first, and the main risks (leakage, imbalance, "
        "skewed target, high-cardinality text)."
    )

    def heuristic(self, ctx: dict) -> CompetitionPlan:
        prof = ctx["profile"]
        sample = prof["sample_submission"]
        id_col = sample["columns"][0]
        target = next((c for c in prof["target_candidates"] if c in sample["columns"]),
                      prof["target_candidates"][0])
        tcol = _col(prof, target)
        numeric = tcol["dtype"].startswith(("int", "float"))
        if not numeric or tcol["n_unique"] <= 2:
            task = "binary" if tcol["n_unique"] <= 2 else "multiclass"
        elif tcol["dtype"].startswith("int") and tcol["n_unique"] <= 20:
            task = "multiclass"
        else:
            task = "regression"

        metric = metric_from_text(ctx.get("evaluation_text", ""))
        sample_target_dtype = sample["dtypes"].get(target, "")
        if metric is None:
            if task == "regression":
                metric = "rmse"
            elif task == "binary" and sample_target_dtype.startswith("float"):
                metric = "roc_auc"
            else:
                metric = "accuracy"

        fmt = self._format(task, metric, sample, target)
        risks = []
        if task == "regression" and (tcol.get("skew") or 0) > 1:
            risks.append(f"target skew {tcol['skew']}: train on log1p(target)")
        text_cols = [c["name"] for c in prof["columns"]
                     if c["dtype"] in ("object", "str", "string") and c["n_unique"] > 0.5 * prof["n_train"]]
        if text_cols:
            risks.append(f"high-cardinality text columns {text_cols}: extract structure or drop")
        return CompetitionPlan(
            reasoning=(f"Target '{target}' is the train-only column present in the sample submission; "
                       f"{tcol['n_unique']} unique values of dtype {tcol['dtype']} -> {task}. "
                       f"Metric '{metric}' from the evaluation text or sample dtype; id '{id_col}' is the "
                       f"first sample column. Gradient boosting first, linear and forests for diversity."),
            task=task, metric=metric, target=target, id_column=id_col, submission_format=fmt,
            candidate_models=(["lightgbm", "hist_gb", "linear"] if prof["n_train"] > LARGE
                              else ["lightgbm", "hist_gb", "linear", "random_forest"]), risks=risks,
        )

    @staticmethod
    def _format(task: str, metric: str, sample: dict, target: str) -> str:
        if task == "regression":
            return "label"
        if metric == "map@3":
            return "top3"
        if task == "multiclass" and len(sample["columns"]) > 2:
            return "multi_proba"
        if task == "binary" and metric in ("roc_auc", "log_loss"):
            return "proba"
        return "label"

    def guard(self, d: CompetitionPlan, ctx: dict) -> tuple[CompetitionPlan, list[str]]:
        prof, notes, upd = ctx["profile"], [], {}
        if d.target not in prof["target_candidates"]:
            upd["target"] = prof["target_candidates"][0]
            notes.append(f"target '{d.target}' not train-only; using '{upd['target']}'")
        if d.id_column not in prof["sample_submission"]["columns"]:
            upd["id_column"] = prof["sample_submission"]["columns"][0]
            notes.append(f"id '{d.id_column}' not in sample; using '{upd['id_column']}'")
        task = upd.get("task", d.task)
        if (task == "regression") != (d.metric in M.REGRESSION_METRICS):
            notes.append(f"metric {d.metric} inconsistent with task {task}; re-deriving")
            fallback = self.heuristic(ctx)
            upd.update(task=fallback.task, metric=fallback.metric, submission_format=fallback.submission_format)
        for key in ("metric", "target", "id_column"):
            override = ctx["overrides"].get(key)
            if override and override != (upd.get(key) or getattr(d, key)):
                upd[key] = override
                notes.append(f"config override {key}={override}")
        if upd.get("metric"):
            upd["submission_format"] = self._format(upd.get("task", d.task), upd["metric"],
                                                    prof["sample_submission"], upd.get("target", d.target))
        return d.model_copy(update=upd), notes

    def _headline(self, d: CompetitionPlan) -> str:
        return f"{d.task} | metric={d.metric} | target={d.target} | format={d.submission_format}"
