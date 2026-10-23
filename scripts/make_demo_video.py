"""Record a short demo video of a finished run: a replay of its event log, then the dashboard.

    python scripts/make_demo_video.py --run runs/evidence/titanic-groq-run7 --out runs/evidence/video

Part 1 replays events.jsonl as a terminal (the same lines the CLI prints while the agents work).
Part 2 opens the Streamlit dashboard on runs/evidence, selects the run and walks through scores,
experiments and the agents' reasoning. Both are recorded with Playwright and joined with ffmpeg
(mp4) when ffmpeg is on PATH; the raw .webm files are kept either way.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOW = {"decision", "tool_result", "fallback"}


def replay_page(run_dir: Path, out: Path) -> Path:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = []
    for e in events:
        if e["event"] not in SHOW and not (e["agent"] == "tools" and e["event"] == "tool_call"):
            continue
        text = f"[it{e['iteration']}] {e['agent']:<10} {e['event']:<11} {e['message']}"
        lines.append(html.escape(text[:220]))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    title = (f"{summary['competition']} | run {summary['run_id']} | LLM {summary['llm']} | "
             f"public {summary.get('best_lb')} | top {summary.get('best_percentile', 0) * 100:.1f}%")
    page = out / "replay.html"
    page.write_text(f"""<!doctype html><meta charset="utf-8"><title>kagent run replay</title>
<style>body{{margin:0;background:#0d1117;color:#c9d1d9;font:15px/1.45 Consolas,monospace}}
header{{padding:14px 20px;background:#161b22;color:#58a6ff;font-weight:bold}}
pre{{margin:0;padding:14px 20px;white-space:pre-wrap}} .research{{color:#d2a8ff}} .evaluator{{color:#7ee787}}
.submit{{color:#f0883e;font-weight:bold}}</style>
<header>kagent: autonomous Kaggle team, replay of {html.escape(title)}</header><pre id="log"></pre>
<script>
const lines = {json.dumps(lines)};
const log = document.getElementById('log'); let i = 0;
function step() {{
  if (i >= lines.length) return;
  const s = document.createElement('span'); const t = lines[i];
  if (t.includes('scout') || t.includes('research')) s.className = 'research';
  if (t.includes('evaluator')) s.className = 'evaluator';
  if (t.includes('public LB') || t.includes('SUBMIT')) s.className = 'submit';
  s.innerHTML = t + '\\n'; log.appendChild(s); window.scrollTo(0, document.body.scrollHeight);
  i++; setTimeout(step, t.includes('cross-validate') || t.includes('train ') ? 60 : 260);
}}
step();
</script>""", encoding="utf-8")
    return page


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/evidence/titanic-groq-run7")
    p.add_argument("--out", default="runs/evidence/video")
    args = p.parse_args()
    from playwright.sync_api import sync_playwright

    run_dir, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    label = f"{summary['competition']} / {summary['run_id']}"
    page_file = replay_page(run_dir, out)

    port = free_port()
    dash = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                             "--server.headless", "true", "--server.port", str(port)],
                            env={**os.environ, "RUNS_DIR": str(ROOT / "runs" / "evidence")},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)

    videos = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            size = {"width": 1280, "height": 720}
            # part 1: replay of the run's event log
            ctx = browser.new_context(viewport=size, record_video_dir=str(out / "raw"), record_video_size=size)
            page = ctx.new_page()
            page.goto(page_file.as_uri())
            page.wait_for_timeout(1000 + 300 * 75)
            videos.append(page.video.path())
            ctx.close()
            # part 2: the dashboard on the same run
            ctx = browser.new_context(viewport=size, record_video_dir=str(out / "raw"), record_video_size=size)
            page = ctx.new_page()
            page.goto(url)
            page.get_by_text("Agent reasoning and decisions").wait_for(timeout=60_000)
            page.wait_for_timeout(2500)
            box = page.get_by_test_id("stSelectbox")
            box.click()
            box.locator("input").fill(summary["run_id"])
            page.get_by_role("option", name=label).click()
            page.get_by_text("Agent reasoning and decisions").wait_for()
            page.wait_for_timeout(4000)
            for _ in range(14):  # scroll through scores, experiments, plots, reasoning timeline
                page.mouse.wheel(0, 450)
                page.wait_for_timeout(1300)
            for text in (" research · decision", " evaluator · reasoning"):  # open the agents' own words
                item = page.get_by_test_id("stExpander").filter(has_text=text).first
                if item.count():
                    item.scroll_into_view_if_needed()
                    item.locator("summary").click()
                    page.wait_for_timeout(5000)
            videos.append(page.video.path())
            ctx.close()
            browser.close()
    finally:
        dash.terminate()

    parts = []
    for i, v in enumerate(videos, 1):
        dest = out / f"part{i}.webm"
        shutil.move(str(v), dest)
        parts.append(dest)
    shutil.rmtree(out / "raw", ignore_errors=True)
    if shutil.which("ffmpeg"):
        listing = out / "parts.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(out / "kagent_titanic_run7.mp4")],
                       check=True)
        listing.unlink()
        for part in parts:  # the mp4 holds both parts
            part.unlink()
    print("videos:", [str(p) for p in parts], "| mp4:", (out / "kagent_titanic_run7.mp4").exists())


if __name__ == "__main__":
    main()
