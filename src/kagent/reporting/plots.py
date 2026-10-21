"""Static evidence for a run, written to runs/<comp>/<run_id>/plots/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..tools.metrics import GREATER_IS_BETTER  # noqa: E402


def make_plots(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    exp = pd.read_csv(run_dir / "experiments.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    metric = summary["metric"]
    higher = GREATER_IS_BETTER[metric]
    paths = []

    # 1. CV score of every experiment, in run order, coloured by iteration.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.viridis(exp["iteration"] / max(exp["iteration"].max(), 1))
    ax.bar(range(len(exp)), exp["cv_score"], color=colors)
    ax.set_xticks(range(len(exp)), exp["name"], rotation=60, ha="right", fontsize=8)
    lo, hi = exp["cv_score"].min(), exp["cv_score"].max()
    pad = (hi - lo) * 0.3 or 0.01
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel(f"CV {metric} ({'higher' if higher else 'lower'} is better)")
    ax.set_title(f"{summary['competition']}: experiments by iteration")
    fig.tight_layout()
    paths.append(out / "cv_by_experiment.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    # 2. Best-so-far CV across experiments.
    best = exp["cv_score"].cummax() if higher else exp["cv_score"].cummin()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(exp) + 1), exp["cv_score"], "o", alpha=0.4, label="experiment")
    ax.step(range(1, len(exp) + 1), best, where="post", label="best so far")
    ax.set_xlabel("experiment #")
    ax.set_ylabel(f"CV {metric}")
    ax.legend()
    ax.set_title("Search progress")
    fig.tight_layout()
    paths.append(out / "best_so_far.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    # 3. CV vs public leaderboard for submitted candidates.
    sub_path = run_dir / "submissions.json"
    if sub_path.exists():
        subs = pd.DataFrame(json.loads(sub_path.read_text())).dropna(subset=["lb_score"])
        if not subs.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(subs["cv_score"], subs["lb_score"])
            for _, r in subs.iterrows():
                label = r["candidate"] + (f"\ntop {r['percentile'] * 100:.0f}%" if pd.notna(r.get("percentile")) else "")
                ax.annotate(label, (r["cv_score"], r["lb_score"]), fontsize=7)
            lims = [min(subs["cv_score"].min(), subs["lb_score"].min()), max(subs["cv_score"].max(), subs["lb_score"].max())]
            ax.plot(lims, lims, "--", color="grey", linewidth=1)
            ax.set_xlabel(f"CV {metric}")
            ax.set_ylabel(f"public LB {metric}")
            ax.set_title("CV vs leaderboard")
            fig.tight_layout()
            paths.append(out / "cv_vs_lb.png")
            fig.savefig(paths[-1], dpi=130)
            plt.close(fig)
    return paths
