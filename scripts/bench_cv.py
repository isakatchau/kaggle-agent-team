"""Offline CV benchmark: one feature plan, a fixed model panel, repeated CV. No Kaggle calls.

    python scripts/bench_cv.py --data data/titanic --feature-plan runs/evidence/titanic-groq/feature_plan_v1.json

Prints JSON with each model's CV, fold std and cv_lcb, plus the nested greedy ensemble, so feature
or modelling changes can be compared under exactly the same protocol. The plan (task, target,
metric) comes from the heuristic Planner, so no LLM is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from kagent.agents import PlannerAgent  # noqa: F401  (import order: agents before tools)
from kagent.agents.schemas import FeaturePlan
from kagent.observability.events import RunLogger
from kagent.tools.features import build_features
from kagent.tools.profile import compact_profile, load_competition_files, profile
from kagent.tools.train import PARAM_SPACE, blend, run_cv

PANEL = {
    "lightgbm": {"num_leaves": 16, "learning_rate": 0.03, "n_estimators": 400, "min_child_samples": 20},
    "hist_gb": {},
    "random_forest": {"n_estimators": 500, "max_depth": 7, "min_samples_leaf": 2},
    "linear": {},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/titanic")
    p.add_argument("--feature-plan", help="FeaturePlan JSON; default: the heuristic Data agent's plan")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default=",".join(PANEL))
    args = p.parse_args()

    train, test, sample = load_competition_files(Path(args.data))
    prof = profile(train, test, sample)
    logger = RunLogger.create(Path("runs/_bench"), "bench", verbose=False)
    plan = PlannerAgent(None, logger).decide({"profile": compact_profile(prof), "evaluation_text": "",
                                              "overrides": {}, "available_models": list(PARAM_SPACE)})
    if args.feature_plan:
        fplan = FeaturePlan.model_validate_json(Path(args.feature_plan).read_text(encoding="utf-8"))
    else:
        from kagent.agents import DataAgent
        fplan = DataAgent(None, logger).heuristic({"iteration": 1, "focus": "features", "plan": plan.model_dump(),
                                                   "profile": compact_profile(prof), "previous_feature_plan": None,
                                                   "feature_importance": {}, "reviewer_suggestions": []})
    raw_y = train[plan.target]
    y = raw_y.to_numpy(dtype=float) if plan.task == "regression" else LabelEncoder().fit_transform(raw_y)
    fm = build_features(train, test, plan.target, plan.id_column, fplan)

    out, results = {"metric": plan.metric, "n_features": fm.n_features, "features": list(fm.X_train.columns),
                    "models": {}}, []
    for model in args.models.split(","):
        t0 = time.time()
        r = run_cv(fm, y, plan.task, plan.metric, model, model, PANEL.get(model, {}), args.folds, args.seed,
                   fplan.log_target, None, repeats=args.repeats)
        results.append(r)
        s = r.summary()
        out["models"][model] = {k: s[k] for k in ("cv_score", "cv_std", "cv_lcb")} | {"seconds": round(time.time() - t0, 1)}
    ens = blend(results, y, plan.metric, "ensemble")
    s = ens.summary()
    out["models"]["ensemble"] = {k: s[k] for k in ("cv_score", "cv_std", "cv_lcb")} | {"weights": ens.params["weights"]}
    out["best_lcb"] = max(v["cv_lcb"] for v in out["models"].values())
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
