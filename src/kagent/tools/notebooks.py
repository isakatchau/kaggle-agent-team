"""Public notebooks of a competition, as text for the Research agent.

Notebooks are downloaded and read, never executed. A notebook that reads data from anywhere but the
competition's own files (another dataset, a URL to a .csv) is marked as leaking and left out: on
Titanic that is how perfect scores are made (the full passenger list with outcomes is public).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

READ_CALL = re.compile(r"""(?:read_csv|read\.csv|read_table|read_excel|read_parquet|fread|loadtxt|open)\(\s*['"]([^'"]+)['"]""")
NOTEBOOK_LINK = re.compile(r"kaggle\.com/(?:code/)?([\w-]+)/([\w-]+)")
DATA_URL = re.compile(r"""https?://[^\s'")]+\.(?:csv|tsv|xlsx?|json|zip|parquet|txt)\b""", re.I)


@dataclass
class Notebook:
    ref: str
    title: str = ""
    votes: int = 0
    markdown: str = ""
    code: str = ""
    external_reads: list[str] = field(default_factory=list)

    @property
    def leaks(self) -> bool:
        return bool(self.external_reads)

    def for_prompt(self, max_chars: int = 2200) -> dict:
        """The opening of the prose and of the code, where tutorials state their method."""
        md = self.markdown[: max_chars // 2]
        return {"ref": self.ref, "title": self.title, "votes": self.votes,
                "markdown": md, "code": self.code[: max_chars - len(md)]}

    def linked_notebooks(self) -> list[str]:
        """Refs of other Kaggle notebooks this one points to (its references)."""
        skip = {"competitions", "datasets", "c", "code", "discussion", "docs", "learn", "models"}
        refs = [f"{o}/{s}" for o, s in NOTEBOOK_LINK.findall(self.markdown) if o not in skip]
        return [r for r in dict.fromkeys(refs) if r != self.ref]


def parse_notebook(path: Path) -> tuple[str, str]:
    """(markdown, code) of a .ipynb / .irnb, or the whole file for a script."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in (".ipynb", ".irnb"):
        try:
            cells = json.loads(raw).get("cells", [])
        except json.JSONDecodeError:
            return "", raw
        join = lambda kind: "\n".join("".join(c.get("source", "")) for c in cells if c.get("cell_type") == kind)
        return join("markdown"), join("code")
    return "", raw


def external_reads(code: str, competition: str, competition_files: list[str]) -> list[str]:
    """Data the code reads from outside the competition: other inputs, or data files by URL."""
    allowed = {f.lower() for f in competition_files}
    bad = []
    for p in READ_CALL.findall(code):
        path = p.replace("\\", "/").lower()
        name = path.rsplit("/", 1)[-1]
        # the competition's folder, or the older layout with its files directly under input/
        inside = f"/{competition.lower()}/" in path or "/" not in path or re.search(r"(^|/)input/[^/]+$", path)
        if not (inside and (name in allowed or not name.endswith((".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet")))):
            if name.endswith((".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".zip", ".txt")):
                bad.append(p)
    return sorted(set(bad + DATA_URL.findall(code)))


def load_notebook(folder: Path, ref: str, competition: str, competition_files: list[str],
                  title: str = "", votes: int = 0) -> Notebook | None:
    files = [p for p in folder.iterdir() if p.suffix in (".ipynb", ".irnb", ".py", ".r", ".R", ".Rmd")]
    if not files:
        return None
    markdown, code = parse_notebook(files[0])
    return Notebook(ref=ref, title=title, votes=votes, markdown=markdown, code=code,
                    external_reads=external_reads(code, competition, competition_files))
