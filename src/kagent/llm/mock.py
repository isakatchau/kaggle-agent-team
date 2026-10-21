from __future__ import annotations

from .base import LLMError


class ScriptedLLM:
    """Returns pre-defined answers keyed by a substring of the system prompt.

    Used in tests to exercise the real LLM code path (parsing, validation,
    guardrails) without a network call.
    """

    name = "scripted"

    def __init__(self, answers: dict[str, dict | Exception]):
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        for key, answer in self.answers.items():
            if key in system:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise LLMError("no scripted answer for this agent")
