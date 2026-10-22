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
    ctx = {"metric": "roc_auc", "candidates": [{"name": "a", "cv_score": 0.9, "cv_std": 0.01, "cv_lcb": 0.897}],
           "best_submitted": None, "submissions_left": 0, "min_improvement": 0.001}
    assert EvaluatorAgent(llm, logger).decide(ctx).submit is False


def test_evaluator_requires_cv_improvement(logger):
    llm = ScriptedLLM({"evaluator": {"reasoning": "r", "submit": True, "candidate": "a"}})
    ctx = {"metric": "rmse", "candidates": [{"name": "a", "cv_score": 0.30, "cv_std": 0.01, "cv_lcb": 0.303}],
           "best_submitted": {"cv_score": 0.2999}, "submissions_left": 2, "min_improvement": 0.001}
    assert EvaluatorAgent(llm, logger).decide(ctx).submit is False


def test_evaluator_gates_on_the_pessimistic_bound(logger):
    # Titanic run 1: an ensemble 0.2 points above the submitted CV but with the widest fold spread
    noisy = {"name": "ens", "cv_score": 0.8395, "cv_std": 0.0287, "cv_lcb": 0.8267}
    steady = {"name": "lgbm", "cv_score": 0.8390, "cv_std": 0.0120, "cv_lcb": 0.8359}
    ctx = {"metric": "accuracy", "candidates": [noisy, steady], "best_submitted": {"cv_score": 0.8373},
           "submissions_left": 2, "min_improvement": 0.0005}
    llm = ScriptedLLM({"evaluator": {"reasoning": "r", "submit": True, "candidate": "ens"}})
    assert EvaluatorAgent(llm, logger).decide(ctx).submit is False
    first = EvaluatorAgent(None, logger).decide({**ctx, "best_submitted": None})
    assert first.candidate == "lgbm" and first.submit


def test_reviewer_cannot_repeat_a_focus_that_just_stalled(logger):
    # Titanic run: the LLM asked for "models" three times in a row with zero CV gain
    llm = ScriptedLLM({"reviewer": {"reasoning": "r", "diagnosis": "d", "continue": True,
                                    "next_focus": "models", "suggestions": []}})
    rounds = [{"focus": "features", "best_cv_gain": None}, {"focus": "models", "best_cv_gain": 0.0}]
    ctx = {"percentile": 0.7, "target_percentile": 0.2, "rounds": rounds, "iterations_left": 2}
    assert LeaderboardReviewerAgent(llm, logger).decide(ctx).next_focus == "tuning"
    rounds[-1]["best_cv_gain"] = 0.004  # a lever that is still paying off may be repeated
    assert LeaderboardReviewerAgent(llm, logger).decide(ctx).next_focus == "models"


def test_reviewer_cannot_stop_after_a_single_round_below_target(logger):
    llm = ScriptedLLM({"reviewer": {"reasoning": "r", "diagnosis": "d", "continue": False,
                                    "next_focus": "stop", "suggestions": []}})
    ctx = {"percentile": 0.81, "target_percentile": 0.2, "rounds": [{"focus": "features", "best_cv_gain": None}],
           "iterations_left": 3}
    review = LeaderboardReviewerAgent(llm, logger).decide(ctx)
    assert review.continue_ and review.next_focus == "models"
    ctx["rounds"].append({"focus": "models", "best_cv_gain": 0.0})  # two rounds: a plateau may be real
    assert LeaderboardReviewerAgent(llm, logger).decide(ctx).next_focus == "stop"


def test_reviewer_stops_at_target(logger):
    llm = ScriptedLLM({"reviewer": {"reasoning": "r", "diagnosis": "d", "continue": True,
                                    "next_focus": "tuning", "suggestions": []}})
    ctx = {"percentile": 0.1, "target_percentile": 0.2, "rounds": [], "iterations_left": 3}
    review = LeaderboardReviewerAgent(llm, logger).decide(ctx)
    assert review.continue_ is False and review.next_focus == "stop"


def test_preflight_fails_loudly_instead_of_falling_back():
    from kagent.llm import check_llm

    with pytest.raises(LLMError):
        check_llm(ScriptedLLM({"JSON": LLMError("401 invalid api key")}))
    check_llm(ScriptedLLM({"JSON": {"ok": True}}))


@pytest.mark.parametrize("base_url, name", [
    (None, "openai:m"), ("https://api.groq.com/openai/v1", "groq:m"), ("https://openrouter.ai/api/v1", "openrouter:m"),
])
def test_openai_compatible_client_names_the_provider(monkeypatch, base_url, name):
    from kagent.llm.openai_client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    if base_url:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert OpenAIClient("m").name == name


def _booking_ctx():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    codes = [f"B{i}" for i in rng.integers(0, 150, 600)]  # repeats in train, reappears in test
    train = pd.DataFrame({"id": range(400), "booking": codes[:400], "name": [f"n{i}" for i in range(400)],
                          "region": rng.choice(["a", "b", "c"], 400), "y": rng.integers(0, 2, 400)})
    test = pd.DataFrame({"id": range(400, 600), "booking": codes[400:], "name": [f"m{i}" for i in range(200)],
                         "region": rng.choice(["a", "b", "c"], 200)})
    prof = profile(train, test, pd.DataFrame({"id": test["id"], "y": 0}))
    return {"profile": prof, "plan": {"target": "y", "id_column": "id", "task": "binary", "metric": "accuracy"},
            "iteration": 1, "focus": "features", "feature_importance": {}}


def test_data_heuristic_proposes_group_encoding_only_for_shared_codes(logger):
    from kagent.agents import DataAgent

    plan = DataAgent(None, logger).decide(_booking_ctx())
    assert [g.keys for g in plan.group_target_encoding] == [["booking"]]  # not the unique name, not region


def test_data_guard_drops_group_encoding_on_a_unique_key(logger):
    from kagent.agents import DataAgent

    llm = ScriptedLLM({"the data agent": {"reasoning": "r", "group_target_encoding": [
        {"keys": ["name"], "name": "family"}, {"keys": ["booking", "nope"], "name": "bad"},
        {"keys": ["booking"], "name": "party"}]}})
    plan = DataAgent(llm, logger).decide(_booking_ctx())
    assert [g.name for g in plan.group_target_encoding] == ["party"]


def test_modeling_guard_does_not_double_the_iteration_prefix(logger):
    from kagent.agents import ModelingAgent
    from kagent.agents.schemas import Experiment, ExperimentBatch

    batch = ExperimentBatch(reasoning="", experiments=[
        Experiment(name="it2_lgbm_1", model="lightgbm", params={}),
        Experiment(name="it1_it1_rf_1", model="random_forest", params={}),
    ])
    ctx = {"history": [], "n_rows": 891, "feature_version": 1, "iteration": 2, "max_experiments": 4}
    guarded, _ = ModelingAgent(None, logger).guard(batch, ctx)
    assert [e.name for e in guarded.experiments] == ["it2_lgbm_1", "it2_rf_1"]
