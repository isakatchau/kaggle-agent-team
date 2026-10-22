from __future__ import annotations

from pathlib import Path

from ..config import LLMConfig
from .base import LLMClient, LLMError


def make_llm(cfg: LLMConfig, log_path: Path | None = None, model_env: str = "OPENAI_MODEL") -> LLMClient | None:
    """Return an LLM client, or None for the deterministic heuristic mode."""
    if cfg.provider == "mock":
        return None
    if cfg.provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(cfg.model, cfg.temperature, log_path, model_env=model_env)
    raise LLMError(f"unknown provider {cfg.provider}")


def check_llm(llm: LLMClient) -> None:
    """One tiny JSON call before the run.

    Agents fall back to their heuristics when the LLM fails, which keeps a run alive but would
    silently turn a misconfigured key, base URL or model name into a heuristic-only run.
    """
    answer = llm.complete_json("Reply with a JSON object.", '{"ping": true, "reply_with": {"ok": true}}')
    if not isinstance(answer, dict):
        raise LLMError(f"expected a JSON object, got {answer!r:.200}")


__all__ = ["LLMClient", "LLMError", "check_llm", "make_llm"]
