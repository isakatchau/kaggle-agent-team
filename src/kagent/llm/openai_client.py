from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from .base import LLMError


class OpenAIClient:
    """OpenAI-compatible chat completions in JSON mode. Every call is appended to llm_calls.jsonl.

    OPENAI_BASE_URL points it at any compatible provider (Groq, OpenRouter, a local server).
    """

    def __init__(self, model: str, temperature: float = 0.2, log_path: Path | None = None,
                 model_env: str = "OPENAI_MODEL"):
        from openai import OpenAI  # imported lazily so the mock path needs no key

        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set (use --llm mock to run without it)")
        self.client = OpenAI(max_retries=5)  # backs off on 429, which free tiers hit often
        self.model = os.getenv(model_env) or os.getenv("OPENAI_MODEL", model)
        # free tiers cap output tokens per minute; an explicit ceiling keeps requests under it
        cap = os.getenv(f"{model_env.replace('OPENAI_MODEL', 'OPENAI_MAX_TOKENS')}")
        try:
            self.max_tokens = int(cap) if cap else None
        except ValueError as exc:
            raise LLMError(f"{model_env.replace('OPENAI_MODEL', 'OPENAI_MAX_TOKENS')} must be an integer") from exc
        # reasoning models spend output tokens thinking before the JSON; providers that support it can cap that
        self.reasoning_effort = os.getenv(model_env.replace("OPENAI_MODEL", "OPENAI_REASONING_EFFORT"))
        self.temperature = temperature
        self.log_path = log_path
        host = urlparse(os.getenv("OPENAI_BASE_URL", "")).hostname or "api.openai.com"
        parts = host.split(".")
        label = host if len(parts) < 2 or host.replace(".", "").isdigit() else parts[-2]  # localhost, 127.0.0.1
        self.name = f"{label}:{self.model}"  # openai:gpt-4o-mini, groq:openai/gpt-oss-120b

    def complete_json(self, system: str, user: str) -> dict:
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
                **({"extra_body": {"reasoning_effort": self.reasoning_effort}} if self.reasoning_effort else {}),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # network, auth, rate limit
            raise LLMError(f"{self.name} call failed: {exc}") from exc

        text = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        self._log(system, user, text, time.time() - t0, usage)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {text[:200]}") from exc

    def _log(self, system, user, text, latency, usage) -> None:
        if not self.log_path:
            return
        record = {
            "model": self.model,
            "latency_s": round(latency, 2),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "system": system,
            "user": user,
            "response": text,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
