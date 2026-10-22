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
        "the CV score against its uncertainty and the CV-to-leaderboard gap seen in past submissions. "
        "`cv_lcb` is the CV moved one standard error towards worse: rank candidates by it, since a "
        "higher CV with a wide fold spread is often a lucky split. Ensemble CV is nested (weights "
        "chosen on other folds), so it is comparable to single-model CV; `params.in_sample_cv` minus "
        "`cv_score` is how much choosing the weights on the scored rows would have inflated it. "
        "Hard rules enforced after you answer: no submission once the budget is spent, and no "
        "submission unless the candidate's `cv_lcb` beats the best submitted CV by `min_improvement`."
    )

    def heuristic(self, ctx: dict) -> SubmitDecision:
        metric = ctx["metric"]
        best = max(ctx["candidates"], key=lambda c: c["cv_lcb"] if M.GREATER_IS_BETTER[metric] else -c["cv_lcb"])
        prev = ctx["best_submitted"]
        improves = M.is_better(metric, best["cv_lcb"], prev["cv_score"] if prev else None, ctx["min_improvement"])
        submit = improves and ctx["submissions_left"] > 0
        why = (f"Most reliable candidate {best['name']}: CV {best['cv_score']:.5f}, "
               f"one standard error worse {best['cv_lcb']:.5f} (fold std {best['cv_std']:.4f}). "
               + (f"Best submitted CV so far {prev['cv_score']:.5f}. " if prev else "Nothing submitted yet. ")
               + ("Even the pessimistic bound clears it; submit." if submit else "Not worth a submission."))
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
        if d.submit and not M.is_better(metric, cand["cv_lcb"], prev["cv_score"] if prev else None,
                                        ctx["min_improvement"]):
            d = d.model_copy(update={"submit": False})
            notes.append("cv_lcb does not beat the best submitted CV by min_improvement")
        return d, notes

    def _headline(self, d: SubmitDecision) -> str:
        return f"{'SUBMIT' if d.submit else 'hold'} {d.candidate}"
