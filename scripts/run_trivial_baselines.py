#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess


def safe_name(series_id: str) -> str:
    return series_id.replace("/", "_").replace(":", "_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trivial event baselines for every trace in a manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        output = args.output_dir / f"{safe_name(row['series_id'])}.csv"
        cmd = [
            "ptaf", "trivial-baselines",
            "--data", row["data"],
            "--series-id", row["series_id"],
            "--horizons", "96", "192", "336", "720",
            "--random-trials", str(args.random_trials),
            "--random-seed", str(args.random_seed),
            "--output", str(output),
        ]
        print(f"=== trivial baselines: {row['series_id']} ===", flush=True)
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
