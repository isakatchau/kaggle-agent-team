"""Shared agent mechanics.

An agent = role prompt + output schema + deterministic heuristic + guardrails.

1. Ask the LLM for a JSON answer that matches the schema (one repair retry).
2. If there is no LLM, or it fails twice, fall back to the heuristic.
3. Apply guardrails (hard rules the LLM cannot override).
4. Log the reasoning and the final decision, including which path produced it.
"""

from __future__ import annotations

import json
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.base import LLMClient, LLMError
from ..observability.events import RunLogger

T = TypeVar("T", bound=BaseModel)


class Agent(Generic[T]):
    name: str = "agent"
    role: str = ""
    schema: type[T]

    def __init__(self, llm: LLMClient | None, logger: RunLogger):
        self.llm = llm
        self.logger = logger

    # --- to implement -----------------------------------------------------
    def heuristic(self, ctx: dict) -> T:
        raise NotImplementedError

    def guard(self, decision: T, ctx: dict) -> tuple[T, list[str]]:
        """Return the (possibly corrected) decision and notes on what was changed."""
        return decision, []

    # --- shared -----------------------------------------------------------
    def system_prompt(self) -> str:
        schema = json.dumps(self.schema.model_json_schema(by_alias=True), indent=None)
        return (
            f"You are the {self.name} agent in a team that competes in Kaggle tabular competitions.\n"
            f"{self.role}\n\n"
            "Answer with ONE JSON object that validates against this JSON schema. "
            "Put your step-by-step justification in the `reasoning` field, citing the numbers "
            "you were given. Only reference columns and models that exist in the context.\n"
            f"SCHEMA: {schema}"
        )

    def decide(self, ctx: dict) -> T:
        decision, source = None, "heuristic"
        if self.llm is not None:
            decision, source = self._ask_llm(ctx)
        if decision is None:
            decision = self.heuristic(ctx)
        decision, notes = self.guard(decision, ctx)
        self.logger.log(self.name, "reasoning", decision.reasoning if hasattr(decision, "reasoning") else "",
                        source=source)
        self.logger.log(self.name, "decision", self._headline(decision), source=source,
                        decision=decision, guardrails=notes)
        return decision

    def _ask_llm(self, ctx: dict) -> tuple[T | None, str]:
        system = self.system_prompt()
        ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}  # "_" keys are for guards, not the prompt
        user = json.dumps(ctx, default=str)
        for attempt in (1, 2):
            try:
                raw = self.llm.complete_json(system, user)
                return self.schema.model_validate(raw), self.llm.name
            except ValidationError as exc:
                self.logger.log(self.name, "error", f"invalid LLM answer (attempt {attempt})",
                                error=str(exc)[:800])
                user = json.dumps({"context": ctx, "previous_answer_errors": str(exc)[:1500],
                                   "instruction": "Fix the errors and answer again."}, default=str)
            except LLMError as exc:
                self.logger.log(self.name, "error", f"LLM call failed (attempt {attempt})", error=str(exc))
        self.logger.log(self.name, "fallback", "using deterministic heuristic")
        return None, "heuristic-fallback"

    def _headline(self, decision: T) -> str:
        return ""
