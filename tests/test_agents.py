import pytest

from kagent.agents import EvaluatorAgent, LeaderboardReviewerAgent, PlannerAgent
from kagent.agents.planner import metric_from_text
from kagent.llm.base import LLMError
from kagent.llm.mock import ScriptedLLM
from kagent.observability.events import RunLogger, read_events
from kagent.tools.profile import load_competition_files, profile


@pytest.fixture
def logger(tmp_path):
    return RunLogger.create(tmp_path, "t", verbose=False)


@pytest.fixture
def planner_ctx(binary_comp):
    train, test, sample = load_competition_files(binary_comp)
    return {"profile": profile(train, test, sample), "evaluation_text": "area under the ROC curve",
            "overrides": {}}


@pytest.mark.parametrize("text,metric", [
    ("Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value", "rmsle"),
    ("percentage of passengers you correctly predict. This is known as accuracy", "accuracy"),
    ("evaluated on area under the ROC curve", "roc_auc"),
    ("Mean Average Precision @ 3 (MAP@3)", "map@3"),
    ("multi-class logarithmic loss", "log_loss"),
])
def test_metric_from_text(text, metric):
    assert metric_from_text(text) == metric


def test_planner_heuristic(planner_ctx, logger):
    plan = PlannerAgent(None, logger).decide(planner_ctx)
    assert (plan.task, plan.metric, plan.target, plan.id_column, plan.submission_format) == \
        ("binary", "roc_auc", "churned", "id", "proba")


def test_planner_guard_corrects_llm_hallucination(planner_ctx, logger):
    llm = ScriptedLLM({"planner": {"reasoning": "r", "task": "binary", "metric": "rmse", "target": "income",
                                   "id_column": "row_id", "submission_format": "label",
                                   "candidate_models": ["lightgbm"]}})
    plan = PlannerAgent(llm, logger).decide(planner_ctx)
    assert plan.target == "churned" and plan.id_column == "id" and plan.metric == "roc_auc"
    decision = [e for e in read_events(logger.run_dir) if e["event"] == "decision"][-1]
    assert decision["data"]["source"] == "scripted" and decision["data"]["guardrails"]


def test_invalid_llm_answers_fall_back_to_heuristic(planner_ctx, logger):
    plan = PlannerAgent(ScriptedLLM({"planner": {"not": "a plan"}}), logger).decide(planner_ctx)
    assert plan.target == "churned"
    events = read_events(logger.run_dir)
    assert sum(e["event"] == "error" for e in events) == 2
    assert any(e["event"] == "fallback" for e in events)


def test_llm_outage_falls_back(planner_ctx, logger):
    plan = PlannerAgent(ScriptedLLM({"planner": LLMError("timeout")}), logger).decide(planner_ctx)
    assert plan.metric == "roc_auc"


def test_evaluator_cannot_exceed_budget(logger):
    llm = ScriptedLLM({"evaluator": {"reasoning": "yolo", "submit": True, "candidate": "a"}})
    ctx = {"metric": "roc_auc", "candidates": [{"name": "a", "cv_score": 0.9, "cv_std": 0.01}],
           "best_submitted": None, "submissions_left": 0, "min_improvement": 0.001}
    assert EvaluatorAgent(llm, logger).decide(ctx).submit is False


def test_evaluator_requires_cv_improvement(logger):
    llm = ScriptedLLM({"evaluator": {"reasoning": "r", "submit": True, "candidate": "a"}})
    ctx = {"metric": "rmse", "candidates": [{"name": "a", "cv_score": 0.30, "cv_std": 0.01}],
           "best_submitted": {"cv_score": 0.2999}, "submissions_left": 2, "min_improvement": 0.001}
    assert EvaluatorAgent(llm, logger).decide(ctx).submit is False


def test_reviewer_stops_at_target(logger):
    llm = ScriptedLLM({"reviewer": {"reasoning": "r", "diagnosis": "d", "continue": True,
                                    "next_focus": "tuning", "suggestions": []}})
    ctx = {"percentile": 0.1, "target_percentile": 0.2, "rounds": [], "iterations_left": 3}
    review = LeaderboardReviewerAgent(llm, logger).decide(ctx)
    assert review.continue_ is False and review.next_focus == "stop"
