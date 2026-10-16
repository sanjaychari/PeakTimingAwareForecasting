#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a controlled breadth study from a trace manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tuning-max-steps", type=int, default=100)
    parser.add_argument("--horizons", default="96 192 336 720")
    parser.add_argument("--seeds", default="1 2 3")
    parser.add_argument(
        "--univariate-models",
        default="dlinear timemixer",
        help="Common low-cost model subset for the breadth study.",
    )
    parser.add_argument(
        "--multivariate-models",
        default="itransformer timemixer",
    )
    parser.add_argument("--losses", default="mse mep")
    args = parser.parse_args()

    sweep = Path(__file__).resolve().parent / "run_horizon_sweep.sh"
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("Manifest contains no traces")

    for row in rows:
        mode = row["mode"]
        models = (
            args.multivariate_models if mode == "multivariate"
            else args.univariate_models
        )
        env = os.environ.copy()
        env.update(
            {
                "HORIZONS": args.horizons,
                "SEEDS": args.seeds,
                "LOSSES": args.losses,
                "MODELS": models,
                "TUNING_MAX_STEPS": str(args.tuning_max_steps),
                "OUTPUT_ROOT": env.get("OUTPUT_ROOT", "results/breadth"),
                "SKIP_EXISTING": env.get("SKIP_EXISTING", "1"),
            }
        )
        cmd = [
            str(sweep),
            row["data"],
            row["series_id"],
            str(args.max_steps),
            mode,
        ]
        if mode == "multivariate":
            cmd.append(row["signals_csv"])
        print(
            f"=== breadth trace={row['series_id']} mode={mode} "
            f"models={models} seeds={args.seeds} ===",
            flush=True,
        )
        subprocess.run(cmd, env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
