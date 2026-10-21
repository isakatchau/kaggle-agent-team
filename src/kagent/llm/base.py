"""LLM interface.

Agents only depend on `LLMClient.complete_json`. Any provider that can return a
JSON object for a (system, user) prompt pair can be plugged in.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict:
        """Return the model's answer parsed as a JSON object, or raise LLMError."""
        ...
