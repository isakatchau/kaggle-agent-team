"""Competition platform tools.

`KaggleCLI` talks to kaggle.com through the official CLI (credentials come from
`kaggle auth login`, KAGGLE_API_TOKEN or ~/.kaggle/access_token, never from this repo).
`OfflineKaggle` implements the same interface against a local folder with a
hidden solution, so the whole loop can be tested without an account.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from . import metrics as M


class PlatformError(RuntimeError):
    pass


class Platform(Protocol):
    name: str

    def download(self, competition: str, dest: Path) -> Path: ...
    def evaluation_text(self, competition: str) -> str: ...
    def submit(self, competition: str, file: Path, message: str) -> float | None: ...
    def leaderboard_scores(self, competition: str, workdir: Path) -> list[float]: ...
    def notebook_list(self, competition: str, sort: str, n: int, search: str | None = None) -> list[dict]: ...
    def notebook_pull(self, ref: str, dest: Path) -> Path | None: ...


def percentile(score: float, board: list[float], metric: str, tol: float = 1e-5) -> float | None:
    """Position of a new submission with `score` as a fraction of the leaderboard (0.05 = top 5%).

    Kaggle ranks equal scores by who reached them first, so a new submission lands behind every team
    already at its score: ties count as ahead. The downloaded board truncates scores to 5 decimals,
    hence the tolerance. (Counting only strictly better teams put Titanic's 327/418 at 22.2% instead
    of the 27.4% the leaderboard shows.)"""
    if not board:
        return None
    arr = np.asarray(board, dtype=float)
    better = arr > score + tol if M.GREATER_IS_BETTER[metric] else arr < score - tol
    ahead = (better | (np.abs(arr - score) <= tol)).sum()
    return float((ahead + 1) / (len(arr) + 1))


# --------------------------------------------------------------------------- real Kaggle
class KaggleCLI:
    name = "kaggle"

    def __init__(self, poll_seconds: int = 10, poll_timeout: int = 300):
        self.poll_seconds = poll_seconds
        self.poll_timeout = poll_timeout

    def _run(self, *args: str, check: bool = True) -> str:
        # current interpreter: works without an activated venv; kaggle_entry trusts the OS certificate store
        cmd = [sys.executable, "-m", "kagent.tools.kaggle_entry", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # notebook titles are unicode
        if check and proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            if "403" in msg or "rules" in msg.lower():
                msg += "\nHint: open the competition page and accept its rules with your account first."
            raise PlatformError(f"kaggle {' '.join(args)} failed: {msg}")
        return proc.stdout

    def download(self, competition: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        if list(dest.glob("*train*.csv")):
            return dest
        self._run("competitions", "download", competition, "-p", str(dest), "-q")
        for z in dest.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()
        return dest

    def evaluation_text(self, competition: str) -> str:
        for args in (
            ("competitions", "pages", competition, "--content", "--page-name", "evaluation", "list"),
            ("competitions", "pages", competition, "list", "--content", "--page-name", "evaluation"),
        ):
            try:
                out = self._run(*args)
                if out.strip():
                    return out[:4000]
            except PlatformError:
                continue
        return ""

    def submit(self, competition: str, file: Path, message: str) -> float | None:
        self._run("competitions", "submit", competition, "-f", str(file), "-m", message, "-q")
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            time.sleep(self.poll_seconds)
            out = self._run("competitions", "submissions", competition, "--csv", check=False)
            score = self._latest_public_score(out)
            if score is not None:
                return score
        return None

    @staticmethod
    def _latest_public_score(csv_text: str) -> float | None:
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            return None
        if df.empty:
            return None
        cols = {c.lower().replace("_", ""): c for c in df.columns}
        col = cols.get("publicscore")
        if not col:
            return None
        value = pd.to_numeric(df.iloc[0][col], errors="coerce")
        return None if pd.isna(value) else float(value)

    def notebook_list(self, competition: str, sort: str, n: int, search: str | None = None) -> list[dict]:
        """Public notebooks of the competition: [{ref, title, votes}]."""
        args = ["kernels", "list", "--competition", competition, "--sort-by", sort, "--page-size", str(n), "--csv"]
        if search:
            args += ["-s", search]
        try:
            out = self._run(*args)
        except PlatformError:
            return []
        try:
            df = pd.read_csv(io.StringIO(out))
        except Exception:
            return []
        return [{"ref": r.ref, "title": str(r.title), "votes": int(r.totalVotes)} for r in df.itertuples()
                if isinstance(r.ref, str)]

    def notebook_pull(self, ref: str, dest: Path) -> Path | None:
        """Download a notebook's source (never run it)."""
        dest.mkdir(parents=True, exist_ok=True)
        try:
            self._run("kernels", "pull", ref, "-p", str(dest))
        except PlatformError:
            return None
        return dest

    def leaderboard_scores(self, competition: str, workdir: Path) -> list[float]:
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            self._run("competitions", "leaderboard", competition, "-d", "-p", str(workdir), "-q")
        except PlatformError:
            return []
        frames = []
        for z in workdir.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".csv"):
                        frames.append(pd.read_csv(zf.open(n)))
        frames += [pd.read_csv(p) for p in workdir.glob("*leaderboard*.csv")]
        for df in frames:
            col = next((c for c in df.columns if c.lower() == "score"), None)
            if col:
                return pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        return []


# --------------------------------------------------------------------------- offline
class OfflineKaggle:
    """A local competition: train/test/sample plus `_solution.csv`, `_leaderboard.csv`, `_meta.json`."""

    name = "offline"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "_meta.json").read_text())

    def download(self, competition: str, dest: Path) -> Path:
        return self.root

    def evaluation_text(self, competition: str) -> str:
        return self.meta.get("evaluation", "")

    def submit(self, competition: str, file: Path, message: str) -> float | None:
        sub = pd.read_csv(file)
        sol = pd.read_csv(self.root / "_solution.csv")
        id_col, target = self.meta["id_column"], self.meta["target"]
        merged = sol.merge(sub, on=id_col, suffixes=("_true", "_pred"))
        if len(merged) != len(sol):
            raise PlatformError("submission ids do not match the test set")
        y, p = merged[f"{target}_true"].to_numpy(), merged[f"{target}_pred"].to_numpy()
        metric = self.meta["metric"]
        if metric == "accuracy":
            score = float((y == p).mean())
        elif metric == "f1":
            score = float(skm.f1_score(y, p))
        else:
            score = M.score(metric, y, p.astype(float))
        with (self.root / "_submissions.csv").open("a") as f:
            f.write(f"{time.time():.0f},{file.name},{score:.6f},{message!r}\n")
        return score

    def notebook_list(self, competition: str, sort: str, n: int, search: str | None = None) -> list[dict]:
        return []  # a local competition has no public notebooks

    def notebook_pull(self, ref: str, dest: Path) -> Path | None:
        return None

    def leaderboard_scores(self, competition: str, workdir: Path) -> list[float]:
        board = self.root / "_leaderboard.csv"
        return pd.read_csv(board)["score"].tolist() if board.exists() else []
