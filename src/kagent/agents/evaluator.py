from __future__ import annotations

from ..tools import metrics as M
from .base import Agent
from .schemas import SubmitDecision


class EvaluatorAgent(Agent[SubmitDecision]):
    name = "evaluator"
    schema = SubmitDecision
    role = (
        "Choose which candidate (single experiment or ensemble) is the most trustworthy for the "
        "hidden test set and decide whether it deserves one of the limited daily submissions. Weigh "
        "the CV score against its fold standard deviation and the CV-to-leaderboard gap seen in past "
        "submissions. Ensemble CV is computed on the same out-of-fold predictions used to pick its "
        "weights, so it is mildly optimistic and has no fold std. Hard rules enforced after you answer: no submission once the budget is spent, "
        "and no submission unless CV beats the best submitted CV by at least `min_improvement`."
    )

    def heuristic(self, ctx: dict) -> SubmitDecision:
        metric = ctx["metric"]
        best = max(ctx["candidates"], key=lambda c: c["cv_score"] if M.GREATER_IS_BETTER[metric] else -c["cv_score"])
        prev = ctx["best_submitted"]
        improves = M.is_better(metric, best["cv_score"], prev["cv_score"] if prev else None, ctx["min_improvement"])
        submit = improves and ctx["submissions_left"] > 0
        std = f"fold std {best['cv_std']:.4f}" if best.get("cv_std") is not None else "ensemble, no fold std"
        why = (f"Best CV candidate {best['name']} = {best['cv_score']:.5f} ({std}). "
               + (f"Best submitted CV so far {prev['cv_score']:.5f}. " if prev else "Nothing submitted yet. ")
               + ("Improvement clears the threshold; submit." if submit else "Not worth a submission."))
        return SubmitDecision(reasoning=why, submit=submit, candidate=best["name"])

    def guard(self, d: SubmitDecision, ctx: dict) -> tuple[SubmitDecision, list[str]]:
        notes, metric = [], ctx["metric"]
        names = {c["name"]: c for c in ctx["candidates"]}
        if d.candidate not in names:
            fallback = self.heuristic(ctx)
            notes.append(f"unknown candidate '{d.candidate}', using {fallback.candidate}")
            d = d.model_copy(update={"candidate": fallback.candidate})
        cand, prev = names[d.candidate], ctx["best_submitted"]
        if d.submit and ctx["submissions_left"] <= 0:
            d = d.model_copy(update={"submit": False})
            notes.append("submission budget exhausted")
        if d.submit and not M.is_better(metric, cand["cv_score"], prev["cv_score"] if prev else None,
                                        ctx["min_improvement"]):
            d = d.model_copy(update={"submit": False})
            notes.append("CV does not beat the best submitted CV by min_improvement")
        return d, notes

    def _headline(self, d: SubmitDecision) -> str:
        return f"{'SUBMIT' if d.submit else 'hold'} {d.candidate}"
