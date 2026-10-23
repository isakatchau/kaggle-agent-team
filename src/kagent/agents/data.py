from __future__ import annotations

import itertools

import pandas as pd

from ..tools.features import validate_plan
from .base import Agent
from .schemas import FeaturePlan

TEXT_DTYPES = ("object", "str", "string")


def _looks_like_date(examples: list[str]) -> bool:
    if not examples:
        return False
    parsed = pd.to_datetime(pd.Series(examples), errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8 and any(ch in examples[0] for ch in "-/")


def _shared_group_key(c: dict, n: int) -> bool:
    """High-cardinality text whose values repeat in train and reappear in test (a ticket, booking or
    household code): a per-group target statistic learned on train can reach the test rows."""
    filled = n * (1 - c["missing_pct"] / 100)
    return (c["dtype"] in TEXT_DTYPES and c["n_unique"] > max(20, 0.05 * n)
            and c["n_unique"] <= 0.9 * filled and c.get("test_seen_pct", 0) >= 10)


class DataAgent(Agent[FeaturePlan]):
    name = "data"
    schema = FeaturePlan
    role = (
        "Design the feature plan for the next modeling round. You do not write code: you choose "
        "transformations from the schema and a tool applies them. Use the profile, the plan's risks, "
        "the feature importances and the reviewer's suggestions from earlier rounds. Typical moves: "
        "extract structure from high-cardinality text with a regex capture group (titles, prefixes, "
        "codes), parse dates, frequency-encode high-cardinality categoricals, group sizes "
        "(e.g. people sharing a ticket), group target encoding when rows that share a code (or a "
        "regex-extracted part of one) plausibly share the outcome and the code reappears in test "
        "(profile test_seen_pct), ratios between meaningful numerics, log-transform skewed "
        "positive numerics, train on log1p(target) for skewed regression targets or RMSLE. Drop "
        "raw columns that are identifiers or free text once their signal is extracted."
    )

    def heuristic(self, ctx: dict) -> FeaturePlan:
        prof, plan = ctx["profile"], ctx["plan"]
        n = prof["n_train"]
        excluded = {plan["target"], plan["id_column"]}
        cols = [c for c in prof["columns"] if c["name"] not in excluded and c["in_test"]]

        drop, text_len, regex, dates, freq, logs, encodings = [], [], [], [], [], [], []
        for c in cols:
            is_text = c["dtype"] in TEXT_DTYPES
            if is_text and _looks_like_date(c["examples"]):
                dates.append(c["name"])
                continue
            if _shared_group_key(c, n):
                encodings.append({"keys": [c["name"]], "name": f"{c['name']}_grp"})
            if is_text and c["n_unique"] > 0.5 * n:
                text_len.append(c["name"])
                regex.append({"column": c["name"], "pattern": r"^\s*([A-Za-z]+)", "name": f"{c['name']}_prefix"})
                drop.append(c["name"])
            elif is_text and c["n_unique"] > 10:
                freq.append(c["name"])
            elif not is_text and (c.get("skew") or 0) > 2 and (c.get("min") or 0) >= 0:
                logs.append(c["name"])

        tcol = next((c for c in prof["columns"] if c["name"] == plan["target"]), {})
        log_target = plan["task"] == "regression" and (
            plan["metric"] == "rmsle" or ((tcol.get("skew") or 0) > 1 and (tcol.get("min") or 0) >= 0))

        ratios, groups, reason_extra = [], [], ""
        if ctx.get("focus") == "features" and ctx.get("feature_importance"):
            numeric = {c["name"] for c in cols if c["dtype"] not in TEXT_DTYPES}
            top = [f for f in ctx["feature_importance"] if f in numeric][:4]
            ratios = [list(p) for p in itertools.combinations(top, 2)]
            cats = [c["name"] for c in cols if c["dtype"] in TEXT_DTYPES and 2 < c["n_unique"] <= 50]
            groups = [cats[:2]] if len(cats) >= 2 else []
            reason_extra = f" Reviewer asked for features: ratios among top numerics {top}, group size {groups}."

        return FeaturePlan(
            reasoning=(f"High-cardinality text {text_len}: keep length and leading token, drop raw. "
                       f"Dates {dates} -> year/month/weekday. Frequency-encode {freq}. log1p skewed {logs}. "
                       f"Group target encoding on codes that repeat and reappear in test "
                       f"{[e['keys'][0] for e in encodings]}. log_target={log_target}." + reason_extra),
            drop_columns=drop, text_length_columns=text_len, regex_features=regex, datetime_columns=dates,
            frequency_encode_columns=freq, log_transform_columns=logs, ratio_features=ratios,
            group_size_columns=groups, group_target_encoding=encodings, log_target=log_target,
        )

    def guard(self, d: FeaturePlan, ctx: dict) -> tuple[FeaturePlan, list[str]]:
        prof, plan = ctx["profile"], ctx["plan"]
        excluded = {plan["target"], plan["id_column"]}
        columns = [c["name"] for c in prof["columns"] if c["in_test"] and c["name"] not in excluded]
        numeric = [c["name"] for c in prof["columns"] if c["dtype"] not in TEXT_DTYPES and c["name"] in columns]
        d, notes = validate_plan(d, columns, numeric)
        kept = set(columns) - set(d.drop_columns)
        if not kept and not d.regex_features and not d.text_length_columns:
            d = d.model_copy(update={"drop_columns": []})
            notes.append("plan dropped every column; ignoring drops")
        if plan["metric"] == "rmsle" and not d.log_target:
            d = d.model_copy(update={"log_target": True})
            notes.append("metric is RMSLE: training on log1p(target)")
        if plan["task"] != "regression" and d.log_target:
            d = d.model_copy(update={"log_target": False})
            notes.append("log_target only applies to regression")

        info = {c["name"]: c for c in prof["columns"]}

        def unique_key(col: str) -> bool:  # never repeats in train, (almost) never reappears in test
            c = info.get(col)
            return c is not None and c.get("test_seen_pct", 100) < 1 and \
                c["n_unique"] >= prof["n_train"] * (1 - c["missing_pct"] / 100)

        useless = [g.name for g in d.group_target_encoding if any(unique_key(k) for k in g.keys)]
        if useless:
            d = d.model_copy(update={"group_target_encoding": [g for g in d.group_target_encoding
                                                               if g.name not in useless]})
            notes.append(f"group_target_encoding {useless}: a key is unique per row, so no group has other members")
        return d, notes

    def _headline(self, d: FeaturePlan) -> str:
        return (f"drop={len(d.drop_columns)} regex={len(d.regex_features)} ratios={len(d.ratio_features)} "
                f"groups={len(d.group_size_columns)} group_te={len(d.group_target_encoding)} "
                f"dates={len(d.datetime_columns)} log_target={d.log_target}")
