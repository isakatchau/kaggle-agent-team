"""LangGraph orchestration.

    planner -> data -> modeling -> evaluator -> reviewer --(features)--> data
                                                       \\--(models|tuning)--> modeling
                                                        \\--(stop)--> finalize -> END

Graph state carries only control flow (iteration, focus). DataFrames, feature
matrices and trained models live in the `Workspace`, which nodes share.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.preprocessing import LabelEncoder

from ..agents import DataAgent, EvaluatorAgent, LeaderboardReviewerAgent, ModelingAgent, PlannerAgent
from ..agents.schemas import CompetitionPlan, Experiment, FeaturePlan
from ..config import RunConfig
from ..llm.base import LLMClient
from ..observability.events import RunLogger, append_index, read_events
from ..tools import metrics as M
from ..tools.features import FeatureMatrix, build_features
from ..tools.kaggle import Platform, PlatformError, percentile
from ..tools.profile import compact_profile, load_competition_files, profile
from ..tools.submission import check_submission, write_submission
from ..tools.train import (PARAM_SPACE, SIGNIFICANCE, ExperimentResult, blend, clip_params, paired_p_better,
                           run_cv)


class GraphState(TypedDict):
    iteration: int
    focus: str
    done: bool


@dataclass
class Workspace:
    cfg: RunConfig
    platform: Platform
    logger: RunLogger
    llm: LLMClient | None
    submit_enabled: bool = True
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    sample: pd.DataFrame | None = None
    profile: dict = field(default_factory=dict)
    evaluation_text: str = ""
    plan: CompetitionPlan | None = None
    feature_plan: FeaturePlan | None = None
    features: FeatureMatrix | None = None
    feature_version: int = 0
    y: np.ndarray | None = None
    classes: np.ndarray | None = None
    results: list[ExperimentResult] = field(default_factory=list)
    submissions: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    leaderboard: list[float] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ helpers
    @property
    def metric(self) -> str:
        return self.plan.metric

    @property
    def group_override_available(self) -> bool:
        return (self.plan.task == "binary" and self.metric in ("accuracy", "f1")
                and bool(self.features is not None and self.features.target_groups))

    @property
    def cv_repeats(self) -> int:
        if self.cfg.cv_repeats is not None:
            return self.cfg.cv_repeats
        return 3 if len(self.train) < 5000 else 1

    def sort_key(self, r: ExperimentResult) -> float:
        return -r.cv_score if M.GREATER_IS_BETTER[self.metric] else r.cv_score

    def best_result(self) -> ExperimentResult | None:
        return min(self.results, key=self.sort_key) if self.results else None

    def best_importance(self) -> dict[str, float]:
        """Feature importance of the best model that exposes one (tree models)."""
        with_imp = [r for r in self.results if r.feature_importance]
        return min(with_imp, key=self.sort_key).feature_importance if with_imp else {}

    def best_submission(self) -> dict | None:
        scored = [s for s in self.submissions if s.get("cv_score") is not None]
        if not scored:
            return None
        sign = -1 if M.GREATER_IS_BETTER[self.metric] else 1
        return min(scored, key=lambda s: sign * s["cv_score"])

    def best_lb(self) -> float | None:
        """Best public score; best_submission() ranks by CV, which is what the submit gate needs."""
        values = [s["lb_score"] for s in self.submissions if s.get("lb_score") is not None]
        if not values:
            return None
        return max(values) if M.GREATER_IS_BETTER[self.metric] else min(values)

    def best_percentile(self) -> float | None:
        values = [s["percentile"] for s in self.submissions if s.get("percentile") is not None]
        return min(values) if values else None

    def write_experiments(self) -> None:
        rows = [r.summary() for r in self.results]
        pd.DataFrame(rows).to_csv(self.logger.run_dir / "experiments.csv", index=False)


# ---------------------------------------------------------------------- nodes
def build_graph(ws: Workspace):
    log = ws.logger
    planner = PlannerAgent(ws.llm, log)
    data_agent = DataAgent(ws.llm, log)
    modeler = ModelingAgent(ws.llm, log)
    evaluator = EvaluatorAgent(ws.llm, log)
    reviewer = LeaderboardReviewerAgent(ws.llm, log)
    cfg = ws.cfg

    def plan_node(state: GraphState) -> dict:
        log.log("tools", "tool_call", f"download {cfg.competition}")
        data_dir = ws.platform.download(cfg.competition, cfg.data_dir / cfg.competition)
        ws.train, ws.test, ws.sample = load_competition_files(data_dir)
        if cfg.sample_rows and len(ws.train) > cfg.sample_rows:
            ws.train = ws.train.sample(cfg.sample_rows, random_state=cfg.seed).reset_index(drop=True)
        ws.profile = profile(ws.train, ws.test, ws.sample)
        ws.evaluation_text = ws.platform.evaluation_text(cfg.competition)
        log.log("tools", "tool_result", f"train={ws.train.shape} test={ws.test.shape}",
                data_dir=data_dir, evaluation_text=ws.evaluation_text[:500])

        overrides = {"metric": cfg.metric, "target": cfg.target, "id_column": cfg.id_column}
        ws.plan = planner.decide({"profile": compact_profile(ws.profile), "evaluation_text": ws.evaluation_text,
                                  "overrides": overrides, "available_models": list(PARAM_SPACE)})
        log.save_json("plan.json", ws.plan)

        raw_y = ws.train[ws.plan.target]
        if ws.plan.task == "regression":
            ws.y = raw_y.to_numpy(dtype=float)
        else:
            enc = LabelEncoder().fit(raw_y)
            ws.y, ws.classes = enc.transform(raw_y), enc.classes_
        return {"iteration": 1, "focus": "features"}

    def data_node(state: GraphState) -> dict:
        log.iteration = state["iteration"]
        best = ws.best_result()
        ctx = {
            "iteration": state["iteration"], "focus": state["focus"],
            "plan": ws.plan.model_dump(), "profile": compact_profile(ws.profile),
            "previous_feature_plan": ws.feature_plan.model_dump() if ws.feature_plan else None,
            "feature_importance": ws.best_importance(),
            "reviewer_suggestions": ws.suggestions,
        }
        previous = (ws.feature_plan, ws.features)
        ws.feature_plan = data_agent.decide(ctx)
        log.log("tools", "tool_call", "build_features")
        try:
            ws.features = build_features(ws.train, ws.test, ws.plan.target, ws.plan.id_column, ws.feature_plan)
        except Exception as exc:  # a bad plan must not end the run
            log.log("tools", "error", f"build_features failed: {exc}")
            if previous[1] is not None:
                ws.feature_plan, ws.features = previous
                log.log("data", "fallback", "keeping the previous feature set")
                return {}
            ws.feature_plan = data_agent.heuristic(ctx)
            ws.features = build_features(ws.train, ws.test, ws.plan.target, ws.plan.id_column, ws.feature_plan)
            log.log("data", "fallback", "using the heuristic feature plan")
        for note in ws.features.skipped:
            log.log("tools", "error", note)
        if ws.feature_plan.group_target_encoding and not ws.features.target_groups:
            # every proposed group encoding was rejected (e.g. category keys): use the heuristic's keys,
            # the same way an unknown candidate falls back to the heuristic choice
            keys = data_agent.heuristic(ctx).group_target_encoding
            if keys:
                ws.feature_plan = ws.feature_plan.model_copy(update={"group_target_encoding": keys})
                ws.features = build_features(ws.train, ws.test, ws.plan.target, ws.plan.id_column,
                                             ws.feature_plan)
                log.log("data", "fallback", f"group target encoding keys replaced by the heuristic's: "
                        f"{[g.keys for g in keys]}")
        ws.feature_version += 1
        log.save_json(f"feature_plan_v{ws.feature_version}.json", ws.feature_plan)
        log.log("tools", "tool_result", f"{ws.features.n_features} features (v{ws.feature_version})",
                applied=ws.features.applied, categorical=ws.features.categorical)
        return {}

    def modeling_node(state: GraphState) -> dict:
        log.iteration = state["iteration"]
        best_before = ws.best_result()
        history = [r.summary() for r in ws.results]
        ctx = {
            "iteration": state["iteration"], "focus": state["focus"], "task": ws.plan.task,
            "metric": ws.metric, "greater_is_better": M.GREATER_IS_BETTER[ws.metric],
            "n_rows": len(ws.train), "n_features": ws.features.n_features,
            "candidate_models": ws.plan.candidate_models, "param_space": PARAM_SPACE,
            "history": history[-20:], "feature_version": ws.feature_version,
            "max_experiments": cfg.budget.max_experiments_per_iteration,
            "reviewer_suggestions": ws.suggestions,
            "group_override_available": ws.group_override_available,
        }
        batch = modeler.decide(ctx)
        defaults = clip_params("group_override", {})[0]

        def is_default(params: dict) -> bool:
            return {k: params.get(k) for k in defaults} == defaults

        if ws.group_override_available and not any(
                r.model == "group_override" and r.feature_version == ws.feature_version and is_default(r.params)
                for r in ws.results) \
                and not any(e.model == "group_override" and is_default(clip_params("group_override", e.params)[0])
                            for e in batch.experiments):
            # like the ensemble, the simplest credible model, at its CV-chosen defaults, is always a
            # candidate on a new feature set (the LLM's own variants run as well)
            batch.experiments.append(Experiment(name=f"it{state['iteration']}_group_override",
                                                model="group_override"))
            log.log("modeling", "decision", "added group_override at its defaults: the simplest candidate")
        for exp in batch.experiments:
            log.log("tools", "tool_call", f"train {exp.name} ({exp.model})", params=exp.params)
            try:
                res = run_cv(ws.features, ws.y, ws.plan.task, ws.metric, exp.name, exp.model, exp.params,
                             cfg.cv_folds, cfg.seed, ws.feature_plan.log_target, log.run_dir / "models",
                             repeats=ws.cv_repeats)
            except Exception as exc:  # one bad experiment must not end the run
                log.log("tools", "error", f"{exp.name} failed: {exc}")
                continue
            res.iteration, res.feature_version = state["iteration"], ws.feature_version
            ws.results.append(res)
            log.log("tools", "tool_result", f"{exp.name} cv={res.cv_score:.5f} ±{np.std(res.fold_scores):.4f} "
                    f"in {res.seconds:.0f}s", result=res.summary(),
                    top_features=dict(list(res.feature_importance.items())[:8]))

        singles = [r for r in ws.results if r.model != "ensemble"]
        if batch.ensemble_top_k >= 2 and len(singles) >= 2:
            # Greedy selection over the best 2k single experiments (at least 4).
            pool = sorted(singles, key=ws.sort_key)[: max(batch.ensemble_top_k * 2, 4)]
            ens = blend(pool, ws.y, ws.metric, f"it{state['iteration']}_ensemble")
            ens.iteration, ens.feature_version = state["iteration"], ws.feature_version
            ws.results.append(ens)
            log.log("tools", "tool_result", f"{ens.name} cv={ens.cv_score:.5f}", weights=ens.params["weights"])

        ws.write_experiments()
        best_after = ws.best_result()
        gain = None
        if best_before is not None and best_after is not None:
            gain = (best_after.cv_score - best_before.cv_score) * (1 if M.GREATER_IS_BETTER[ws.metric] else -1)
        ws.rounds.append({"iteration": state["iteration"], "focus": state["focus"],
                          "feature_version": ws.feature_version,
                          "n_experiments": len(batch.experiments),
                          "best_cv": best_after.cv_score if best_after else None,
                          "best_cv_gain": gain})
        return {}

    def evaluator_node(state: GraphState) -> dict:
        log.iteration = state["iteration"]
        ranked = sorted(ws.results, key=ws.sort_key)
        lead = ranked[0]
        p_lead = {r.name: paired_p_better(lead, r, ws.y, ws.metric) for r in ranked}
        # the top 8 plus every candidate the best does not significantly beat, so the simplest is never cut
        if all(p is None for p in p_lead.values()):
            tied = [r for r in ranked[8:] if -ws.sort_key(r) >= -ws.sort_key(lead) - lead.cv_se1]
        else:
            tied = [r for r in ranked[8:] if (p_lead[r.name] or 0.0) >= SIGNIFICANCE]
        top = ranked[:8] + tied
        prev = ws.best_submission()
        ctx = {
            "metric": ws.metric, "greater_is_better": M.GREATER_IS_BETTER[ws.metric],
            "candidates": [r.summary() | ({"p_best_better": round(p_lead[r.name], 4)}
                                          if p_lead[r.name] is not None else {}) for r in top],
            "best_submitted": prev,
            "past_submissions": ws.submissions,
            "submissions_left": cfg.budget.max_submissions - len(ws.submissions),
            "min_improvement": cfg.budget.min_improvement,
        }
        decision = evaluator.decide(ctx)
        cand = next(r for r in ws.results if r.name == decision.candidate)
        path = log.run_dir / "submissions" / f"{cand.name}.csv"
        write_submission(path, ws.sample, ws.test[ws.plan.id_column], cand.test_pred,
                         ws.plan.submission_format, ws.classes, ws.plan.id_column)
        problems = check_submission(path, ws.sample)
        if problems:
            log.log("evaluator", "error", "submission failed pre-flight checks", problems=problems)
            return {}
        if not (decision.submit and ws.submit_enabled):
            log.log("evaluator", "decision", f"saved {path.name} without submitting",
                    submit_enabled=ws.submit_enabled)
            ws.rounds[-1].update(submitted=False)
            return {}

        log.log("tools", "tool_call", f"submit {path.name}")
        try:
            lb = ws.platform.submit(cfg.competition, path, f"kagent {log.run_id} {cand.name} cv={cand.cv_score:.5f}")
        except PlatformError as exc:
            log.log("tools", "error", f"submit failed: {exc}")
            return {}
        if not ws.leaderboard:
            ws.leaderboard = ws.platform.leaderboard_scores(cfg.competition, log.run_dir / "leaderboard")
        pct = percentile(lb, ws.leaderboard, ws.metric) if lb is not None else None
        sub = {"iteration": state["iteration"], "candidate": cand.name, "cv_score": cand.cv_score,
               "lb_score": lb, "percentile": pct, "file": str(path), "leaderboard_size": len(ws.leaderboard)}
        ws.submissions.append(sub)
        ws.rounds[-1].update(submitted=True, lb_score=lb, percentile=pct)
        log.save_json("submissions.json", ws.submissions)
        pct_txt = f"top {pct * 100:.1f}%" if pct is not None else "percentile unknown"
        log.log("tools", "tool_result", f"public LB {lb} ({pct_txt} of {len(ws.leaderboard)})", submission=sub)
        return {}

    def reviewer_node(state: GraphState) -> dict:
        log.iteration = state["iteration"]
        best = ws.best_result()
        ctx = {
            "metric": ws.metric, "greater_is_better": M.GREATER_IS_BETTER[ws.metric],
            "target_percentile": cfg.target_percentile, "percentile": ws.best_percentile(),
            "rounds": ws.rounds, "submissions": ws.submissions,
            "iterations_left": cfg.budget.max_iterations - state["iteration"],
            "submissions_left": cfg.budget.max_submissions - len(ws.submissions),
            "best_experiment": best.summary() if best else None,
            "top_features": dict(list(ws.best_importance().items())[:10]),
            "feature_plan": ws.feature_plan.model_dump(),
            "leaderboard_quantiles": (np.quantile(ws.leaderboard, [0.05, 0.2, 0.5]).round(5).tolist()
                                      if ws.leaderboard else None),
        }
        review = reviewer.decide(ctx)
        ws.suggestions = review.suggestions
        log.save_json(f"review_it{state['iteration']}.json", review)
        if not review.continue_:
            return {"done": True}
        return {"iteration": state["iteration"] + 1, "focus": review.next_focus, "done": False}

    def prune_checkpoints(best: ExperimentResult | None) -> None:
        if cfg.keep_checkpoints == "all" or best is None:
            return
        keep = set(best.params.get("weights", {})) if best.model == "ensemble" else {best.name}
        for f in (log.run_dir / "models").glob("*.joblib"):
            if f.stem not in keep:
                f.unlink()
        log.save_json("models/best_candidate.json", {"best": best.name, "checkpoints": sorted(keep),
                                                     "params": best.params})

    def finalize_node(state: GraphState) -> dict:
        best = ws.best_result()
        prune_checkpoints(best)
        summary = {
            "run_id": log.run_id, "competition": cfg.competition, "platform": ws.platform.name,
            "llm": ws.llm.name if ws.llm else "heuristic",
            "llm_fallbacks": sum(e["event"] == "fallback" for e in read_events(log.run_dir)),
            "metric": ws.metric,
            "iterations": state["iteration"], "experiments": len(ws.results),
            "best_cv": best.cv_score if best else None, "best_experiment": best.name if best else None,
            "submissions": len(ws.submissions),
            "best_lb": ws.best_lb(),
            "best_percentile": ws.best_percentile(), "target_percentile": cfg.target_percentile,
            "minutes": round((time.time() - ws.started) / 60, 1), "run_dir": str(log.run_dir),
        }
        log.save_json("summary.json", summary)
        append_index(cfg.runs_dir, summary)
        try:
            from ..reporting.plots import make_plots

            make_plots(log.run_dir)
        except Exception as exc:
            log.log("reporting", "error", f"plots failed: {exc}")
        log.log("orchestrator", "decision", "run finished", summary=summary)
        return {"done": True}

    def route(state: GraphState) -> str:
        if state.get("done"):
            return "finalize"
        return "data" if state["focus"] == "features" else "modeling"

    g = StateGraph(GraphState)
    for name, fn in [("planner", plan_node), ("data", data_node), ("modeling", modeling_node),
                     ("evaluator", evaluator_node), ("reviewer", reviewer_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "data")
    g.add_edge("data", "modeling")
    g.add_edge("modeling", "evaluator")
    g.add_edge("evaluator", "reviewer")
    g.add_conditional_edges("reviewer", route, {"data": "data", "modeling": "modeling", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run(ws: Workspace) -> dict:
    graph = build_graph(ws)
    ws.logger.log("orchestrator", "decision", f"start {ws.cfg.competition} on {ws.platform.name}",
                  config=ws.cfg.model_dump(mode="json"), llm=ws.llm.name if ws.llm else "heuristic")
    graph.invoke({"iteration": 0, "focus": "features", "done": False},
                 {"recursion_limit": 6 * ws.cfg.budget.max_iterations + 10})
    return json.loads((ws.logger.run_dir / "summary.json").read_text())
