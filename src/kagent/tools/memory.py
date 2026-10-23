"""Research memory: rules a competition's research has already validated, kept across runs.

A person who has read a competition's tutorials does not reread them before every submission. The
research of each run adds its valid rules here; later runs revalidate them on the data and evaluate
them as candidates, so one unlucky translation by the language model does not lose what an earlier
run found. Selection between remembered and new rules is left to cross-validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..agents.schemas import RuleModel


def rule_key(rule: RuleModel) -> str:
    """Identity of a rule's content, ignoring its name and provenance."""
    return json.dumps(rule.model_dump(exclude={"name", "source", "reported_public_score"}), sort_keys=True)


def load_memory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("rules", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def remember(path: Path, rules: list[RuleModel], found_in: str, model: str) -> list[dict]:
    """Add rules not already remembered; returns the updated memory."""
    memory = load_memory(path)
    known = {rule_key(RuleModel.model_validate(m["rule"])) for m in memory}
    for rule in rules:
        if rule_key(rule) not in known:
            known.add(rule_key(rule))
            memory.append({"rule": rule.model_dump(), "found_in": found_in, "model": model})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rules": memory}, indent=2), encoding="utf-8")
    return memory


def forget(path: Path, rules: list[RuleModel]) -> None:
    """Drop rules that failed to run, so later runs do not retry them."""
    if not path.exists():
        return
    drop = {rule_key(r) for r in rules}
    memory = [m for m in load_memory(path) if rule_key(RuleModel.model_validate(m["rule"])) not in drop]
    path.write_text(json.dumps({"rules": memory}, indent=2), encoding="utf-8")


def safe_name(name: str) -> str:
    """A name written by a language model, made safe for file names and logs."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")[:40] or "rule"


def unique_names(rules: list[RuleModel]) -> list[RuleModel]:
    """Distinct contents with distinct names (rules are evaluated and selected by name)."""
    seen, names, out = set(), set(), []
    for rule in rules:
        key = rule_key(rule)
        if key in seen:
            continue
        seen.add(key)
        name = rule.name
        while name in names:
            name += "_"
        names.add(name)
        out.append(rule.model_copy(update={"name": name}))
    return out
