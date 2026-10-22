"""Command line entry point.

    python -m kagent run --config configs/competitions/titanic.yaml
    python -m kagent demo                      # offline, no keys needed
    python -m kagent list-playground
    python -m kagent kaggle auth login         # kaggle CLI with the OS certificate store
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import truststore
from dotenv import load_dotenv

from .config import RunConfig
from .llm import LLMError, check_llm, make_llm
from .observability.events import RunLogger
from .orchestrator.graph import Workspace, run
from .tools.kaggle import KaggleCLI, OfflineKaggle
from .tools.synthetic import make_competition


def _execute(cfg: RunConfig, platform, submit: bool, quiet: bool) -> dict:
    logger = RunLogger.create(cfg.runs_dir, cfg.competition, verbose=not quiet)
    try:
        llm = make_llm(cfg.llm, logger.run_dir / "llm_calls.jsonl")
        if llm is not None:
            check_llm(llm)
    except LLMError as exc:
        sys.exit(f"error: {exc}\nCheck OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL in .env, "
                 "or run with --llm mock.")
    research_llm = None
    if llm is not None and os.getenv("OPENAI_MODEL_RESEARCH"):
        # reading notebooks is the largest prompt; a separate model keeps it off the other agents' quota
        try:
            research_llm = make_llm(cfg.llm, logger.run_dir / "llm_calls.jsonl", model_env="OPENAI_MODEL_RESEARCH")
            check_llm(research_llm)
        except LLMError as exc:
            sys.exit(f"error: research model: {exc}")
    ws = Workspace(cfg=cfg, platform=platform, logger=logger, llm=llm, submit_enabled=submit,
                   research_llm=research_llm)
    summary = run(ws)
    print("\n" + json.dumps(summary, indent=2))
    return summary


def cmd_run(args) -> None:
    load_dotenv()
    cfg = RunConfig.from_yaml(args.config)
    if args.llm:
        cfg.llm.provider = args.llm
    if args.max_iterations:
        cfg.budget.max_iterations = args.max_iterations
    if args.target_percentile:
        cfg.target_percentile = args.target_percentile
    if args.offline:
        platform = OfflineKaggle(Path(args.offline))
    else:
        platform = KaggleCLI()
    _execute(cfg, platform, submit=not args.no_submit, quiet=args.quiet)


def cmd_demo(args) -> None:
    load_dotenv()
    root = Path(args.data) / f"synthetic-{args.task}"
    if not (root / "_meta.json").exists():
        make_competition(root, args.task)
    cfg = RunConfig(competition=f"synthetic-{args.task}", data_dir=Path(args.data),
                    runs_dir=Path(args.runs), cv_folds=5)
    cfg.llm.provider = args.llm
    cfg.budget.max_iterations = args.max_iterations
    cfg.target_percentile = args.target_percentile
    _execute(cfg, OfflineKaggle(root), submit=True, quiet=args.quiet)


def _kaggle(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "kagent.tools.kaggle_entry", *args], check=False).returncode


def cmd_list(args) -> None:
    _kaggle("competitions", "list", "--category", "playground")


def cmd_kaggle(args) -> None:
    sys.exit(_kaggle(*args.kaggle_args))


def main(argv: list[str] | None = None) -> None:
    # verify TLS against the OS store: TLS-inspecting antivirus and corporate proxies put their root there
    truststore.inject_into_ssl()
    p = argparse.ArgumentParser(prog="kagent", description="Multi-agent Kaggle team")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the team on a competition config")
    r.add_argument("--config", required=True)
    r.add_argument("--llm", choices=["openai", "mock"], help="override llm.provider")
    r.add_argument("--max-iterations", type=int)
    r.add_argument("--target-percentile", type=float, help="stop once the LB percentile is at or below this")
    r.add_argument("--offline", help="folder with a local competition (see `demo`) instead of kaggle.com")
    r.add_argument("--no-submit", action="store_true", help="write submission files but never upload")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("demo", help="offline end-to-end run on a generated competition")
    d.add_argument("--task", choices=["binary", "regression"], default="binary")
    d.add_argument("--llm", choices=["openai", "mock"], default="mock")
    d.add_argument("--max-iterations", type=int, default=3)
    d.add_argument("--target-percentile", type=float, default=0.20)
    d.add_argument("--data", default="data")
    d.add_argument("--runs", default="runs")
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_demo)

    sub.add_parser("list-playground", help="list Kaggle playground competitions").set_defaults(func=cmd_list)

    k = sub.add_parser("kaggle", help="the official kaggle CLI, e.g. `kagent kaggle auth login`")
    k.add_argument("kaggle_args", nargs=argparse.REMAINDER)
    k.set_defaults(func=cmd_kaggle)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
