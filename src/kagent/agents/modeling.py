from __future__ import annotations

import json
import re

from ..tools.train import PARAM_SPACE, clip_params
from .base import Agent
from .schemas import Experiment, ExperimentBatch


class ModelingAgent(Agent[ExperimentBatch]):
    name = "modeling"
    schema = ExperimentBatch
    role = (
        "Propose the next batch of cross-validated experiments. Each experiment is a model family "
        "plus hyperparameters from `param_space` (values outside the bounds are clipped). Use the "
        "history to avoid repeating configurations, to tune around what worked, and to add diverse "
        "models that make a useful ensemble. Small datasets need stronger regularisation (fewer "
        "leaves, more min samples, lower learning rate). Set ensemble_top_k >= 2 to run greedy "
        "ensemble selection over the best 2*k experiments (0 disables it)."
    )

    def heuristic(self, ctx: dict) -> ExperimentBatch:
        focus, it = ctx.get("focus"), ctx["iteration"]
        n_rows, k = ctx["n_rows"], ctx["max_experiments"]
        small = n_rows < 3000
        leaves = 15 if small else 31
        tried = {e["model"] for e in ctx["history"]}

        if it == 1 or focus == "features":
            exps = [Experiment(name="lgbm_base", model="lightgbm",
                               params={"num_leaves": leaves, "min_child_samples": 20 if not small else 10})]
            exps += [Experiment(name=f"{m}_base", model=m) for m in ctx["candidate_models"] if m != "lightgbm"]
            why = "First round on this feature set: one default per candidate family to compare."
        elif focus == "models":
            fresh = [m for m in PARAM_SPACE if m not in tried]
            exps = [Experiment(name=f"{m}_base", model=m) for m in fresh]
            exps.append(Experiment(name="lgbm_reg", model="lightgbm",
                                   params={"num_leaves": max(7, leaves // 2), "min_child_samples": 40,
                                           "colsample_bytree": 0.5, "reg_lambda": 5.0}))
            why = f"Reviewer asked for model diversity; untried families {fresh} plus a regularised LightGBM."
        else:
            exps = [
                Experiment(name="lgbm_slow", model="lightgbm",
                           params={"learning_rate": 0.01, "n_estimators": 2000, "num_leaves": leaves}),
                Experiment(name="lgbm_shallow", model="lightgbm",
                           params={"num_leaves": 7, "min_child_samples": 30, "colsample_bytree": 0.5}),
                Experiment(name="lgbm_wide", model="lightgbm",
                           params={"num_leaves": 63 if not small else 31, "min_child_samples": 50,
                                   "subsample": 0.7, "reg_lambda": 3.0}),
                Experiment(name="hgb_tuned", model="hist_gb",
                           params={"learning_rate": 0.03, "max_iter": 800, "l2_regularization": 1.0}),
            ]
            why = "Tuning round: slower learning rate, shallower and wider LightGBM variants, tuned HistGB."
        return ExperimentBatch(reasoning=why, experiments=exps[:k], ensemble_top_k=3)

    def guard(self, d: ExperimentBatch, ctx: dict) -> tuple[ExperimentBatch, list[str]]:
        notes, seen, keep = [], set(), []
        done = {(e["model"], json.dumps(e["params"], sort_keys=True), e["feature_version"])
                for e in ctx["history"]}
        for e in d.experiments:
            if ctx["n_rows"] > 200_000 and e.model in ("random_forest", "extra_trees"):
                notes.append(f"skipped {e.name}: forests are too slow above 200k rows")
                continue
            params, _ = clip_params(e.model, e.params)
            sig = (e.model, json.dumps(params, sort_keys=True), ctx["feature_version"])
            if sig in done or sig in seen:
                notes.append(f"skipped duplicate {e.name}")
                continue
            seen.add(sig)
            base = re.sub(r"^(it\d+_)+", "", e.name)  # LLMs often echo the prefix they saw in history
            keep.append(e.model_copy(update={"name": f"it{ctx['iteration']}_{base}"[:60]}))
        if len(keep) > ctx["max_experiments"]:
            notes.append(f"trimmed batch to {ctx['max_experiments']}")
            keep = keep[: ctx["max_experiments"]]
        if not keep:
            notes.append("every proposal was a duplicate; adding a seeded LightGBM variant")
            keep = [Experiment(name=f"it{ctx['iteration']}_lgbm_alt", model="lightgbm",
                               params={"num_leaves": 23, "colsample_bytree": 0.6, "learning_rate": 0.02,
                                       "n_estimators": 1200})]
        return d.model_copy(update={"experiments": keep}), notes

    def _headline(self, d: ExperimentBatch) -> str:
        return f"{[e.name for e in d.experiments]} ensemble_top_k={d.ensemble_top_k}"
