"""Browser test of the Streamlit dashboard against the committed evidence runs.

    make e2e        # records a video per test under runs/evidence/videos
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "runs" / "evidence"
RUN_ID = {c: json.loads((EVIDENCE / c / "summary.json").read_text())["run_id"]
          for c in ("synthetic-binary", "synthetic-regression")}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
         "--server.headless", "true", "--server.port", str(port)],
        env={**os.environ, "RUNS_DIR": str(EVIDENCE)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("streamlit did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


def _open(page, url):
    page.goto(url)
    page.get_by_text("Agent reasoning and decisions").wait_for(timeout=30_000)


def test_attempts_table_lists_every_run(page, dashboard_url):
    _open(page, dashboard_url)
    page.get_by_text("Attempts").wait_for()
    box = page.get_by_test_id("stSelectbox")
    for label in (f"synthetic-binary / {RUN_ID['synthetic-binary']}", f"synthetic-regression / {RUN_ID['synthetic-regression']}"):
        box.click()
        box.locator("input").fill(label.split(" / ")[0])
        assert page.get_by_role("option", name=label).is_visible()
        page.keyboard.press("Escape")


def test_selected_run_shows_scores_and_agent_timeline(page, dashboard_url):
    _open(page, dashboard_url)
    box = page.get_by_test_id("stSelectbox")
    box.click()
    box.locator("input").fill("synthetic-regression")  # the option list is virtualised: filter, then pick
    page.get_by_role("option", name=f"synthetic-regression / {RUN_ID['synthetic-regression']}").click()

    page.get_by_text("Best CV (rmsle)").wait_for(timeout=30_000)  # Streamlit reruns after the selection
    metrics = page.get_by_test_id("stMetric")
    assert "rmsle" in metrics.nth(0).inner_text().lower()
    assert "top 10.0%" in metrics.nth(2).inner_text()

    # four iterations: the reviewer routes each round, the evaluator holds back submissions
    timeline = page.get_by_test_id("stExpander")
    timeline.filter(has_text="it4 · reviewer · decision").wait_for()
    assert timeline.filter(has_text="next_focus=").count() >= 4
    assert timeline.filter(has_text="evaluator · decision").count() >= 4


def test_agent_filter_narrows_timeline(page, dashboard_url):
    _open(page, dashboard_url)
    timeline = page.get_by_test_id("stExpander")
    timeline.filter(has_text=" · planner · ").first.wait_for()
    before = timeline.filter(has_text=" · planner · ").count()
    assert before > 0

    agents = page.get_by_test_id("stMultiSelect").first
    agents.get_by_role("button", name="Remove planner").click()  # chip close button
    page.wait_for_timeout(1500)
    assert timeline.filter(has_text=" · planner · ").count() == 0
