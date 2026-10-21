"""Structured run log.

Every agent message, decision, tool call and LLM exchange is appended to
`runs/<competition>/<run_id>/events.jsonl`, one JSON object per line. The
dashboard and the plots read only from these files, so what you see is exactly
what the agents did.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


@dataclass
class RunLogger:
    run_dir: Path
    run_id: str
    competition: str
    iteration: int = 0
    verbose: bool = True
    _t0: float = field(default_factory=time.time)

    @classmethod
    def create(cls, runs_dir: Path, competition: str, verbose: bool = True) -> "RunLogger":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = runs_dir / competition / run_id
        for sub in ("submissions", "models", "plots"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return cls(run_dir=run_dir, run_id=run_id, competition=competition, verbose=verbose)

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    def log(self, agent: str, event: str, message: str = "", **data: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - self._t0, 2),
            "run_id": self.run_id,
            "competition": self.competition,
            "iteration": self.iteration,
            "agent": agent,
            "event": event,
            "message": message,
            "data": _jsonable(data),
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self.verbose:
            print(f"[{record['elapsed_s']:>7.1f}s] it{self.iteration} {agent:<10} {event:<12} {message}")

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(_jsonable(payload), indent=2))
        return path


def read_events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_index(runs_dir: Path, summary: dict) -> None:
    """One line per finished run, so the dashboard can list attempts across competitions."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "index.jsonl").open("a") as f:
        f.write(json.dumps(_jsonable(summary)) + "\n")
