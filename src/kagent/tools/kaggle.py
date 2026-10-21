"""Competition platform tools.

`KaggleCLI` talks to kaggle.com through the official CLI (credentials come from
~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY, never from this repo).
`OfflineKaggle` implements the same interface against a local folder with a
hidden solution, so the whole loop can be tested without an account.
"""

from __future__ import annotations

import io
import json
import subprocess
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


def percentile(score: float, board: list[float], metric: str) -> float | None:
    """Fraction of the leaderboard at or ahead of `score` (0.05 = top 5%)."""
    if not board:
        return None
    arr = np.asarray(board, dtype=float)
    ahead = (arr > score).sum() if M.GREATER_IS_BETTER[metric] else (arr < score).sum()
    return float((ahead + 1) / (len(arr) + 1))


# --------------------------------------------------------------------------- real Kaggle
class KaggleCLI:
    name = "kaggle"

    def __init__(self, poll_seconds: int = 10, poll_timeout: int = 300):
        self.poll_seconds = poll_seconds
        self.poll_timeout = poll_timeout

    def _run(self, *args: str, check: bool = True) -> str:
        cmd = ["kaggle", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if check and proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            if "403" in msg or "rules" in msg.lower():
                msg += "\nHint: open the competition page and accept its rules with your account first."
            raise PlatformError(f"{' '.join(cmd)} failed: {msg}")
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

    def leaderboard_scores(self, competition: str, workdir: Path) -> list[float]:
        board = self.root / "_leaderboard.csv"
        return pd.read_csv(board)["score"].tolist() if board.exists() else []
