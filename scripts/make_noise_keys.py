"""Noise control for group features: copy a competition and add shuffled copies of key columns.

    python scripts/make_noise_keys.py --data data/titanic --out runs/_noise/titanic_s0 --seed 0 \
        --key NoiseSurname=Name:"^([^,]+)," --key NoiseTicket=Ticket

Each NAME=COLUMN[:REGEX] adds NAME, a random permutation over train+test rows of COLUMN (or of its regex
capture). The permutation keeps the cardinality and the group-size distribution of the real key but breaks
any link between group membership and the target, so a fold-safe group target encoding on NAME must give no
CV gain beyond noise. Run scripts/bench_cv.py on --out with a plan that encodes and drops NAME.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from kagent.tools.profile import load_competition_files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key", action="append", required=True, help="NAME=COLUMN[:REGEX]")
    args = p.parse_args()

    train, test, _ = load_competition_files(Path(args.data))
    rng = np.random.default_rng(args.seed)
    for spec in args.key:
        name, source = spec.split("=", 1)
        column, _, pattern = source.partition(":")
        values = pd.concat([train[column], test[column]], ignore_index=True).astype("string")
        if pattern:
            values = values.str.extract(pattern, expand=False)
        shuffled = values.to_numpy()[rng.permutation(len(values))]
        train[name], test[name] = shuffled[: len(train)], shuffled[len(train):]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    for f in Path(args.data).glob("*submission*.csv"):
        shutil.copy(f, out / f.name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
