"""Browser test of the Streamlit dashboard against the committed evidence runs.

    make e2e        # records a video per test under runs/evidence/videos
"""

from __future__ import annotations

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
    options = page.get_by_test_id("stSelectbox")
    options.click()
    listbox = page.get_by_role("listbox")
    assert listbox.get_by_text("synthetic-binary / 20261021T030536Z").is_visible()
    assert listbox.get_by_text("synthetic-regression / 20261021T030558Z").is_visible()


def test_selected_run_shows_scores_and_agent_timeline(page, dashboard_url):
    _open(page, dashboard_url)
    page.get_by_test_id("stSelectbox").click()
    page.get_by_role("listbox").get_by_text("synthetic-regression / 20261021T030558Z").click()

    metrics = page.get_by_test_id("stMetric")
    metrics.first.wait_for()
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
