"""Rebuild a competition's research memory from logged Research-agent answers.

    python scripts/seed_research_memory.py --competition titanic --data data/titanic --target Survived         --id-column PassengerId runs/evidence/titanic-research/seed_logs

Scans llm_calls.jsonl files for Research-agent responses, keeps the rules that validate on the
competition's data (known columns, regexes that match, conditions that match rows) and adds them to
runs/<competition>/research_memory.json. Nothing is invented here: every rule is a logged answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kagent.agents import ResearchAgent  # noqa: F401  (import order: agents before tools)
from kagent.agents.schemas import RuleModel
from kagent.tools.memory import remember
from kagent.tools.profile import load_competition_files
from kagent.tools.rules import validate_rule


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--competition", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--id-column", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("logs", nargs="+")
    args = p.parse_args()
    train, test, _ = load_competition_files(Path(args.data))
    columns = [c for c in train.columns if c not in (args.target, args.id_column)]
    data = pd.concat([train.drop(columns=[args.target]), test], ignore_index=True)
    files = [f for x in map(Path, args.logs) for f in ([x] if x.is_file() else sorted(x.rglob("llm_calls.jsonl")))]
    added = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            if "research agent" not in call["system"][:80]:
                continue
            try:  # rule by rule: an answer written under an older schema may hold more rules than it allows now
                raw_rules = json.loads(call["response"]).get("rules", [])
            except (json.JSONDecodeError, AttributeError):
                continue
            valid = []
            for raw in raw_rules:
                try:
                    rule = RuleModel.model_validate(raw)
                except Exception:
                    continue
                if not validate_rule(rule, columns, data):
                    valid.append(rule)
            if valid:
                before = len(json.loads((Path(args.runs) / args.competition / "research_memory.json").read_text())["rules"]) \
                    if (Path(args.runs) / args.competition / "research_memory.json").exists() else 0
                memory = remember(Path(args.runs) / args.competition / "research_memory.json", valid,
                                  f"{f.parent.name} (logged answer)", call["model"])
                added += len(memory) - before
    print(f"scanned {len(files)} logs, added {added} rules")


if __name__ == "__main__":
    main()
