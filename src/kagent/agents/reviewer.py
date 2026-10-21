from __future__ import annotations

from .base import Agent
from .schemas import Review


class LeaderboardReviewerAgent(Agent[Review]):
    name = "reviewer"
    schema = Review
    role = (
        "Review where the team stands: CV history per round, public leaderboard scores, the estimated "
        "leaderboard percentile (0.10 = top 10%) against the target, CV-vs-leaderboard gap, and "
        "remaining budget. Diagnose what limits the score (features, model choice, tuning, "
        "overfitting to CV) and propose the next plan: `next_focus` routes the team (features -> Data "
        "agent redesigns features; models / tuning -> Modeling agent; stop -> finish), and "
        "`suggestions` are concrete instructions passed to the next agents. Stop when the target is "
        "reached, improvements have plateaued, or the budget is spent."
    )

    def heuristic(self, ctx: dict) -> Review:
        pct, target = ctx.get("percentile"), ctx["target_percentile"]
        rounds = ctx["rounds"]
        last_focus = rounds[-1]["focus"] if rounds else None
        gains = [r["best_cv_gain"] for r in rounds if r.get("best_cv_gain") is not None]
        if pct is not None and pct <= target:
            return Review(reasoning=f"Percentile {pct:.3f} is within target {target}.", diagnosis="target reached",
                          continue_=False, next_focus="stop")
        if ctx["iterations_left"] <= 0:
            return Review(reasoning="No iterations left.", diagnosis="budget exhausted",
                          continue_=False, next_focus="stop")
        # Rotate: features -> tuning -> models, skipping the area that just stalled.
        order = ["features", "tuning", "models"]
        nxt = order[(order.index(last_focus) + 1) % 3] if last_focus in order else "features"
        stalled = bool(gains) and gains[-1] is not None and gains[-1] <= 0
        pct_txt = f"{pct:.3f}" if pct is not None else "unknown"
        return Review(
            reasoning=(f"Estimated percentile {pct_txt} vs target {target}. Last round focus={last_focus}, "
                       f"CV gain={gains[-1] if gains else 'n/a'}. Next focus: {nxt}."),
            diagnosis="last change stalled" if stalled else "still improving",
            continue_=True, next_focus=nxt,
            suggestions=[{"features": "add interactions between the most important numeric features and group sizes",
                          "tuning": "tune LightGBM around the best config: lower learning rate, regularise",
                          "models": "add diverse model families for the ensemble"}[nxt]],
        )

    def guard(self, d: Review, ctx: dict) -> tuple[Review, list[str]]:
        notes = []
        pct = ctx.get("percentile")
        if ctx["iterations_left"] <= 0 and d.continue_:
            d = d.model_copy(update={"continue_": False, "next_focus": "stop"})
            notes.append("iteration budget exhausted")
        if pct is not None and pct <= ctx["target_percentile"] and d.continue_:
            d = d.model_copy(update={"continue_": False, "next_focus": "stop"})
            notes.append("target percentile reached")
        if d.continue_ and d.next_focus == "stop":
            d = d.model_copy(update={"continue_": False})
        if not d.continue_ and d.next_focus != "stop":
            d = d.model_copy(update={"next_focus": "stop"})
        return d, notes

    def _headline(self, d: Review) -> str:
        return f"continue={d.continue_} next_focus={d.next_focus} | {d.diagnosis}"
