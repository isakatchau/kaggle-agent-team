from __future__ import annotations

from pathlib import Path

from ..config import LLMConfig
from .base import LLMClient, LLMError


def make_llm(cfg: LLMConfig, log_path: Path | None = None) -> LLMClient | None:
    """Return an LLM client, or None for the deterministic heuristic mode."""
    if cfg.provider == "mock":
        return None
    if cfg.provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(cfg.model, cfg.temperature, log_path)
    raise LLMError(f"unknown provider {cfg.provider}")


__all__ = ["LLMClient", "LLMError", "make_llm"]
