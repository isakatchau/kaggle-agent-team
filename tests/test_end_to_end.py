import json

from kagent.config import RunConfig
from kagent.observability.events import RunLogger, read_events
from kagent.orchestrator.graph import Workspace, run
from kagent.tools.kaggle import OfflineKaggle


def _run(comp_dir, tmp_path, target_percentile=0.01, iterations=2):
    cfg = RunConfig(competition=comp_dir.name, data_dir=comp_dir.parent, runs_dir=tmp_path / "runs",
                    cv_folds=3, target_percentile=target_percentile)
    cfg.llm.provider = "mock"
    cfg.budget.max_iterations = iterations
    cfg.budget.max_experiments_per_iteration = 2
    logger = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=False)
    return run(Workspace(cfg=cfg, platform=OfflineKaggle(comp_dir), logger=logger, llm=None)), logger


def test_binary_loop_runs_and_logs_every_agent(binary_comp, tmp_path):
    summary, logger = _run(binary_comp, tmp_path)
    assert summary["submissions"] >= 1 and summary["best_lb"] is not None
    assert summary["llm_fallbacks"] == 0
    agents = {e["agent"] for e in read_events(logger.run_dir) if e["event"] == "decision"}
    assert {"planner", "data", "modeling", "evaluator", "reviewer"} <= agents
    for f in ("plan.json", "experiments.csv", "summary.json", "plots/cv_by_experiment.png"):
        assert (logger.run_dir / f).exists(), f
    assert list((logger.run_dir / "models").glob("*.joblib"))
    index = (tmp_path / "runs" / "index.jsonl").read_text().splitlines()
    assert json.loads(index[-1])["run_id"] == summary["run_id"]


def test_regression_loop_stops_on_budget(regression_comp, tmp_path):
    summary, _ = _run(regression_comp, tmp_path, iterations=2)
    assert summary["iterations"] == 2 and summary["metric"] == "rmsle"


def test_summary_best_lb_is_the_best_public_score_not_the_best_cv():
    from types import SimpleNamespace

    subs = [{"cv_score": 0.837, "lb_score": 0.782, "percentile": 0.22},
            {"cv_score": 0.840, "lb_score": 0.770, "percentile": 0.66}]
    ws = SimpleNamespace(submissions=subs, metric="accuracy")
    assert Workspace.best_submission(ws)["lb_score"] == 0.770  # the submit gate compares CV
    assert Workspace.best_lb(ws) == 0.782
    assert Workspace.best_percentile(ws) == 0.22
