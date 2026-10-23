"""Streamlit dashboard: competition attempts, scores and agent decisions.

    streamlit run dashboard/app.py            # reads ./runs
    RUNS_DIR=path/to/runs streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

RUNS = Path(os.getenv("RUNS_DIR", "runs"))
st.set_page_config(page_title="Kaggle agent team", layout="wide")
st.title("Kaggle agent team")


def load_index() -> pd.DataFrame:
    # recursive, keeping the real directory: committed evidence lives in runs/evidence/<competition>/
    rows = []
    for summary in RUNS.rglob("summary.json"):
        rows.append({**json.loads(summary.read_text()), "dir": str(summary.parent)})
    return pd.DataFrame(rows).sort_values("run_id", ascending=False) if rows else pd.DataFrame()


index = load_index()
if index.empty:
    st.info(f"No finished runs in `{RUNS}`. Try `python -m kagent demo`.")
    st.stop()

st.subheader("Attempts")
cols = ["run_id", "competition", "platform", "llm", "metric", "iterations", "experiments",
        "best_cv", "submissions", "best_lb", "best_percentile", "target_percentile", "minutes"]
st.dataframe(index[[c for c in cols if c in index]], hide_index=True, width="stretch")

dirs = {f"{r.competition} / {r.run_id}": Path(r.dir) for r in index.itertuples()}
run_dir = dirs[st.selectbox("Run", list(dirs))]
summary = json.loads((run_dir / "summary.json").read_text())

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Best CV ({summary['metric']})", f"{summary['best_cv']:.5f}" if summary["best_cv"] is not None else "-")
c2.metric("Best public LB", f"{summary['best_lb']:.5f}" if summary.get("best_lb") is not None else "-")
pct = summary.get("best_percentile")
c3.metric("Leaderboard position", f"top {pct * 100:.1f}%" if pct is not None else "unknown",
          help=f"target: top {summary['target_percentile'] * 100:.0f}%")
c4.metric("Experiments / submissions", f"{summary['experiments']} / {summary['submissions']}")

left, right = st.columns([3, 2])
exp = pd.read_csv(run_dir / "experiments.csv")
with left:
    st.subheader("Experiments")
    try:  # keep run order instead of alphabetical
        st.bar_chart(exp, x="name", y="cv_score", sort=False)
    except TypeError:
        st.bar_chart(exp.set_index("name")["cv_score"])
    st.dataframe(exp[["iteration", "name", "model", "cv_score", "cv_std", "seconds", "feature_version"]],
                 hide_index=True, width="stretch")
with right:
    st.subheader("Submissions")
    sub_path = run_dir / "submissions.json"
    if sub_path.exists():
        st.dataframe(pd.DataFrame(json.loads(sub_path.read_text()))[
            ["iteration", "candidate", "cv_score", "lb_score", "percentile"]], hide_index=True)
    else:
        st.write("No submissions.")
    for png in sorted((run_dir / "plots").glob("*.png")):
        st.image(str(png), caption=png.stem)

st.subheader("Agent reasoning and decisions")
events = pd.DataFrame([json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()])
agents = st.multiselect("Agents", sorted(events["agent"].unique()), default=sorted(events["agent"].unique()))
kinds = st.multiselect("Events", sorted(events["event"].unique()),
                       default=[k for k in ("reasoning", "decision", "error", "fallback") if k in set(events["event"])])
view = events[events["agent"].isin(agents) & events["event"].isin(kinds)]
for e in view.itertuples():
    with st.expander(f"it{e.iteration} · {e.agent} · {e.event} · {e.message[:140]}"):
        st.write(e.message)
        st.json(e.data, expanded=False)

with st.expander("Competition plan"):
    st.json(json.loads((run_dir / "plan.json").read_text()))
for fp in sorted(run_dir.glob("feature_plan_v*.json")):
    with st.expander(fp.stem):
        st.json(json.loads(fp.read_text()))
