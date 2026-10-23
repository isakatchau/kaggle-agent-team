import json

import numpy as np
import pandas as pd

from kagent.agents import NotebookScoutAgent, ResearchAgent
from kagent.agents.schemas import RuleModel
from kagent.config import RunConfig
from kagent.llm.mock import ScriptedLLM
from kagent.observability.events import RunLogger, read_events
from kagent.orchestrator.graph import Workspace, run
from kagent.tools.kaggle import OfflineKaggle
from kagent.tools.notebooks import external_reads
from kagent.tools.rules import _frame, predict_rule, run_rule_cv, validate_rule

FILES = ["train.csv", "test.csv", "sample_submission.csv"]


def test_external_reads_flags_data_from_outside_the_competition():
    code = """
train = pd.read_csv('../input/comp/train.csv')
test = pd.read_csv("/kaggle/input/comp/test.csv")
leak = pd.read_csv('../input/full-answers/answers.csv')
url = pd.read_csv("https://example.org/raw/master/answers.csv")
pics = open('../input/comp/notes.md')
"""
    assert external_reads(code, "comp", FILES) == ["../input/full-answers/answers.csv",
                                                  "https://example.org/raw/master/answers.csv"]


def _households(seed=0, n_groups=150):
    # base feature explains most outcomes; households share fate, which overrides flip correctly
    rng = np.random.default_rng(seed)
    hh = np.repeat(np.arange(n_groups), 4)
    kind = rng.choice(["a", "b"], len(hh), p=[0.6, 0.4])
    fate = rng.integers(0, 2, n_groups)[hh]
    y = np.where(kind == "a", np.where(rng.random(len(hh)) < 0.75, fate, 1 - fate), (rng.random(len(hh)) < 0.2).astype(int))
    df = pd.DataFrame({"id": range(len(hh)), "kind": kind, "name": [f"Fam{h}, X" for h in hh], "y": y})
    cut = int(len(df) * 0.7)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].drop(columns="y").reset_index(drop=True)


RULE = {"name": "household", "base_column": "kind",
        "derive": [{"column": "name", "pattern": "^([^,]+),", "name": "family"}],
        "overrides": [{"when": [{"column": "kind", "op": "==", "value": "a"}], "group_keys": ["family"],
                       "members_when": [[{"column": "kind", "op": "==", "value": "a"}]], "if_all": 0, "set_to": 0},
                      {"when": [{"column": "kind", "op": "==", "value": "a"}], "group_keys": ["family"],
                       "members_when": [[{"column": "kind", "op": "==", "value": "a"}]], "if_all": 1, "set_to": 1}]}


def test_rule_never_reads_labels_of_rows_it_predicts():
    train, test = _households()
    rule = RuleModel.model_validate(RULE)
    full = _frame(rule, train.drop(columns="y"), test)
    y = np.concatenate([train["y"].to_numpy(), np.zeros(len(test), dtype=int)])
    known, rows = np.arange(0, 300), np.arange(300, len(train))
    flipped = y.copy()
    flipped[rows] = 1 - flipped[rows]
    assert (predict_rule(rule, full, y, known, rows) == predict_rule(rule, full, flipped, known, rows)).all()


def test_group_overrides_add_signal_and_validation_catches_bad_rules():
    train, test = _households()
    y = train["y"].to_numpy()
    base = RuleModel.model_validate({**RULE, "overrides": []})
    full = RuleModel.model_validate(RULE)
    a = run_rule_cv(base, train.drop(columns="y"), test, y, "accuracy", "base", 5, 0)
    b = run_rule_cv(full, train.drop(columns="y"), test, y, "accuracy", "rule", 5, 0)
    assert b.cv_score > a.cv_score + 0.03
    bad = RuleModel.model_validate({**RULE, "base_column": "nope"})
    assert validate_rule(bad, ["id", "kind", "name"]) == ["unknown column nope"]


def test_research_guard_keeps_only_valid_rules_on_binary_tasks(tmp_path):
    logger = RunLogger.create(tmp_path, "t", verbose=False)
    notes = {"reasoning": "r", "techniques": [], "rules": [RULE, {**RULE, "name": "bad", "base_column": "nope"}]}
    agent = ResearchAgent(ScriptedLLM({"research": notes}), logger)
    ctx = {"task": "binary", "columns": ["id", "kind", "name"], "notebooks": []}
    assert [r.name for r in agent.decide(ctx).rules] == ["household"]
    assert agent.decide({**ctx, "task": "regression"}).rules == []
    scout = NotebookScoutAgent(ScriptedLLM({"scout": {"reasoning": "r", "read": ["a/x", "zz/unknown"]}}), logger)
    assert scout.decide({"listing": [{"ref": "a/x", "list": "votes"}], "max_read": 4}).read == ["a/x"]


class NotebookPlatform(OfflineKaggle):
    """A local competition that also has one public notebook."""

    def notebook_list(self, competition, sort, n, search=None):
        return [{"ref": "someone/household-rule", "title": "Households share fate", "votes": 10}]

    def notebook_pull(self, ref, dest):
        dest.mkdir(parents=True, exist_ok=True)
        nb = {"cells": [{"cell_type": "markdown", "source": ["Group 'a' rows by family; families share fate."]},
                        {"cell_type": "code", "source": ["df = pd.read_csv('../input/hh/train.csv')"]}]}
        (dest / "nb.ipynb").write_text(json.dumps(nb))
        return dest


def test_research_rules_become_cross_validated_candidates(tmp_path):
    train, test = _households(seed=1)
    root = tmp_path / "hh"
    root.mkdir()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    test[["id"]].assign(y=0).to_csv(root / "sample_submission.csv", index=False)
    (root / "_meta.json").write_text(json.dumps({"metric": "accuracy"}))
    cfg = RunConfig(competition="hh", data_dir=tmp_path, runs_dir=tmp_path / "runs", cv_folds=3, metric="accuracy")
    cfg.budget.max_iterations = 1
    cfg.budget.max_experiments_per_iteration = 1
    llm = ScriptedLLM({"scout": {"reasoning": "r", "read": ["someone/household-rule"]},
                       "research": {"reasoning": "r", "techniques": [], "rules": [RULE]}})
    for agent in ("planner", "data", "modeling", "evaluator", "reviewer"):
        llm.answers[f"the {agent} agent"] = __import__("kagent.llm.base", fromlist=["LLMError"]).LLMError("use heuristic")
    logger = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=False)
    run(Workspace(cfg=cfg, platform=NotebookPlatform(root), logger=logger, llm=llm, submit_enabled=False))
    results = pd.read_csv(logger.run_dir / "experiments.csv")
    assert "it1_rule_household" in set(results["name"])
    research = json.loads((logger.run_dir / "research.json").read_text())
    assert research["read"][0]["external_reads"] == [] and research["rules"][0]["name"] == "household"
    assert any(e["agent"] == "research" and e["event"] == "decision" for e in read_events(logger.run_dir))


def test_rule_validation_with_data_catches_translation_errors(tmp_path):
    train, test = _households()
    data = pd.concat([train.drop(columns="y"), test], ignore_index=True)
    wrong = RuleModel.model_validate({**RULE, "overrides": [{**RULE["overrides"][0],
                                      "when": [{"column": "kind", "op": "in", "value": ["woman", "boy"]}]}]})
    issues = validate_rule(wrong, ["id", "kind", "name"], data)
    assert len(issues) == 1 and "no row has that value" in issues[0] and "'a'" in issues[0]
    assert validate_rule(RuleModel.model_validate(RULE), ["id", "kind", "name"], data) == []


def test_research_repairs_rejected_rules_once(tmp_path):
    train, test = _households()
    data = pd.concat([train.drop(columns="y"), test], ignore_index=True)
    bad = {**RULE, "name": "first_try", "overrides": [{**RULE["overrides"][0],
                                                       "when": [{"column": "kind", "op": "==", "value": "woman"}]}]}

    class TwoAnswers(ScriptedLLM):
        def complete_json(self, system, user):
            self.calls.append((system, user))
            ok = "rejected_rules" in user
            assert '"_data"' not in user  # private keys never reach the prompt
            return {"reasoning": "r", "rules": [RULE if ok else bad]}

    logger = RunLogger.create(tmp_path, "t", verbose=False)
    llm = TwoAnswers({})
    notes = ResearchAgent(llm, logger).decide({"task": "binary", "columns": ["id", "kind", "name"], "_data": data})
    assert [r.name for r in notes.rules] == ["household"] and len(llm.calls) == 2


def test_research_keeps_valid_rules_from_several_attempts(tmp_path):
    other = {**RULE, "name": "household", "base_column": "name"}  # same name, different rule

    class Alternating(ScriptedLLM):
        def complete_json(self, system, user):
            self.calls.append((system, user))
            return {"reasoning": "r", "rules": [RULE if len(self.calls) % 2 else other]}

    logger = RunLogger.create(tmp_path, "t", verbose=False)
    llm = Alternating({})
    notes = ResearchAgent(llm, logger).decide({"task": "binary", "columns": ["id", "kind", "name"], "_attempts": 3})
    assert [r.base_column for r in notes.rules] == ["kind", "name"] and len(llm.calls) == 3
    assert len({r.name for r in notes.rules}) == 2


def test_research_memory_carries_rules_to_later_runs_without_research(tmp_path):
    train, test = _households(seed=2)
    root = tmp_path / "hh"
    root.mkdir()
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    test[["id"]].assign(y=0).to_csv(root / "sample_submission.csv", index=False)
    (root / "_meta.json").write_text(json.dumps({"metric": "accuracy"}))
    cfg = RunConfig(competition="hh", data_dir=tmp_path, runs_dir=tmp_path / "runs", cv_folds=3, metric="accuracy")
    cfg.budget.max_iterations = 1
    cfg.budget.max_experiments_per_iteration = 1
    llm = ScriptedLLM({"scout": {"reasoning": "r", "read": ["someone/household-rule"]},
                       "research": {"reasoning": "r", "techniques": [], "rules": [RULE]}})
    for agent in ("planner", "data", "modeling", "evaluator", "reviewer"):
        llm.answers[f"the {agent} agent"] = __import__("kagent.llm.base", fromlist=["LLMError"]).LLMError("heuristic")
    first = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=False)
    run(Workspace(cfg=cfg, platform=NotebookPlatform(root), logger=first, llm=llm, submit_enabled=False))
    assert (cfg.runs_dir / "hh" / "research_memory.json").exists()
    # second run: no LLM and a platform without notebooks; the remembered rule is still a candidate
    second = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=False)
    run(Workspace(cfg=cfg, platform=OfflineKaggle(root), logger=second, llm=None, submit_enabled=False))
    assert "it1_rule_household" in set(pd.read_csv(second.run_dir / "experiments.csv")["name"])


def test_rule_values_are_type_checked_and_bad_values_reported_not_raised():
    train, test = _households()
    data = pd.concat([train.drop(columns="y"), test], ignore_index=True)
    cond = lambda c: RuleModel.model_validate({**RULE, "overrides": [{**RULE["overrides"][0], "when": [c]}]})
    cols = ["id", "kind", "name"]
    assert "needs one value" in validate_rule(cond({"column": "kind", "op": "==", "value": ["a", "b"]}), cols, data)[0]
    assert "needs a number" in validate_rule(cond({"column": "kind", "op": "<", "value": "16"}), cols, data)[0]
    assert "no row has that value" in validate_rule(cond({"column": "kind", "op": "not_in", "value": ["man"]}), cols, data)[0]
    nested = RuleModel.model_validate({**RULE, "derive": [{"column": "name", "pattern": "^((a+)+),", "name": "family"}]})
    assert any("nested quantifiers" in i for i in validate_rule(nested, cols, data))


def test_numeric_in_matches_float_columns_and_members_only_choose_evidence():
    from kagent.tools.rules import _frame, _mask, predict_rule
    df = pd.DataFrame({"age": [30.0, 31.0, 5.0], "kind": ["b", "a", "a"], "fam": ["f", "f", "f"]})
    m = _mask(df, [RuleModel.model_validate({**RULE, "overrides": [{**RULE["overrides"][0], "when": [
        {"column": "age", "op": "not_in", "value": [30, 31]}]}]}).overrides[0].when[0]])
    assert m.tolist() == [False, False, True]
    # a father (kind b, not a member) is set to survive because every member (kind a) of his family survived
    rule = RuleModel.model_validate({"name": "r", "base_column": "kind", "overrides": [{
        "when": [{"column": "kind", "op": "==", "value": "b"}], "group_keys": ["fam"],
        "members_when": [[{"column": "kind", "op": "==", "value": "a"}]], "if_all": 1, "set_to": 1}]})
    y = np.array([0, 1, 1])
    full = _frame(rule, df, df.iloc[:0])
    assert predict_rule(rule, full, y, np.array([1, 2]), np.array([0])).tolist() == [1]


def test_memory_skips_bad_entries_and_rule_names_are_path_safe(tmp_path):
    from kagent.tools.memory import load_memory, safe_name
    assert safe_name("../../data/comp/train") == "data_comp_train"
    assert safe_name("Age<16 boys: cat?") == "Age_16_boys_cat"
    train, test = _households(seed=3)
    root = tmp_path / "hh"
    root.mkdir()
    train.assign(leak=train["y"]).to_csv(root / "train.csv", index=False)  # train-only column recording y
    test.to_csv(root / "test.csv", index=False)
    test[["id"]].assign(y=0).to_csv(root / "sample_submission.csv", index=False)
    (root / "_meta.json").write_text(json.dumps({"metric": "accuracy"}))
    mem = tmp_path / "runs" / "hh" / "research_memory.json"
    mem.parent.mkdir(parents=True)
    leak_rule = {**RULE, "name": "../../oops", "base_column": "leak", "overrides": []}
    stale = {**RULE, "name": "stale", "overrides": [{**RULE["overrides"][0], "min_group_size": 1}]}
    mem.write_text(json.dumps({"rules": [{"rule": RULE}, {"rule": leak_rule}, {"rule": stale}]}))
    cfg = RunConfig(competition="hh", data_dir=tmp_path, runs_dir=tmp_path / "runs", cv_folds=3, metric="accuracy")
    cfg.budget.max_iterations = 1
    cfg.budget.max_experiments_per_iteration = 1
    logger = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=False)
    run(Workspace(cfg=cfg, platform=OfflineKaggle(root), logger=logger, llm=None, submit_enabled=False))
    names = set(pd.read_csv(logger.run_dir / "experiments.csv")["name"])
    assert "it1_rule_household" in names and not any("oops" in n for n in names)  # the leak column is not usable
    assert all(p.parent.name == "submissions" for p in (logger.run_dir / "submissions").glob("*.csv"))
