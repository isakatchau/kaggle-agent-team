from __future__ import annotations

from ..tools import metrics as M
from .base import Agent
from .schemas import SubmitDecision


def simplest_tied(candidates: list[dict], metric: str) -> dict:
    """One-standard-error rule (Breiman et al.): among candidates whose CV is within one standard error
    (`cv_se1`) of the best CV, the one with the lowest `complexity`. CV cannot tell tied candidates
    apart; the simpler one has fewer decisions that can be fitted to noise."""
    sign = 1 if M.GREATER_IS_BETTER[metric] else -1
    best = max(candidates, key=lambda c: sign * c["cv_score"])
    tied = [c for c in candidates if sign * c["cv_score"] >= sign * best["cv_score"] - best["cv_se1"]]
    return min(tied, key=lambda c: (c["complexity"], -sign * c["cv_score"]))


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
        "`complexity` counts a candidate's fitted decisions. Hard rules enforced after you answer: "
        "among candidates within one standard error (`cv_se1`) of the best CV the simplest is used "
        "unless yours beats it by more than that (one-standard-error rule); no submission once the "
        "budget is spent; no submission unless the candidate's `cv_lcb` beats the best submitted CV "
        "by `min_improvement`."
    )

    def heuristic(self, ctx: dict) -> SubmitDecision:
        metric = ctx["metric"]
        best = simplest_tied(ctx["candidates"], metric)
        prev = ctx["best_submitted"]
        improves = M.is_better(metric, best["cv_lcb"], prev["cv_score"] if prev else None, ctx["min_improvement"])
        submit = improves and ctx["submissions_left"] > 0
        why = (f"Simplest candidate within one standard error of the best CV: {best['name']} "
               f"(CV {best['cv_score']:.5f}, complexity {best['complexity']:.0f}, cv_lcb {best['cv_lcb']:.5f}). "
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
        simple = simplest_tied(ctx["candidates"], metric)
        sign = 1 if M.GREATER_IS_BETTER[metric] else -1
        chosen = names[d.candidate]
        if chosen["name"] != simple["name"] and chosen["complexity"] > simple["complexity"] \
                and sign * (chosen["cv_score"] - simple["cv_score"]) <= chosen["cv_se1"]:
            notes.append(f"one-standard-error rule: {d.candidate} (complexity {chosen['complexity']:.0f}) is "
                         f"tied with the simpler {simple['name']} (complexity {simple['complexity']:.0f})")
            d = d.model_copy(update={"candidate": simple["name"]})
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
