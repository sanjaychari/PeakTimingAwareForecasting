from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_trace_manifest(
    frame: pd.DataFrame,
    data_path: Path,
    target_signal: str,
    output: Path,
    *,
    n_traces: int = 3,
    mode: str = "univariate",
    required_signals: tuple[str, ...] = (),
    input_size: int = 336,
    max_horizon: int = 720,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
) -> Path:
    """Select a deterministic breadth-study trace set by longest eligible duration.

    Selection uses only metadata/length and required signal availability, not
    forecasting performance or event rate.
    """
    required = {"unique_id", "signal", "scenario", "ds", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns required for trace selection: {sorted(missing)}")
    if mode not in {"univariate", "multivariate"}:
        raise ValueError("mode must be univariate or multivariate")

    min_n = max(
        int(np.ceil((input_size + max_horizon) / train_fraction)),
        int(np.ceil(max_horizon / val_fraction)),
        int(np.ceil(max_horizon / (1.0 - train_fraction - val_fraction))),
    )

    candidates = []
    for scenario, group in frame.groupby("scenario", sort=True):
        target_rows = group.loc[group["signal"].astype(str).eq(target_signal)].sort_values("ds")
        if target_rows.empty:
            continue
        if len(target_rows) < min_n:
            continue

        signals_csv = ""
        if mode == "multivariate":
            signal_groups = {
                signal: g.sort_values("ds")
                for signal, g in group.groupby("signal", sort=False)
            }
            if any(signal not in signal_groups for signal in required_signals):
                continue
            reference_ds = target_rows["ds"].to_numpy()
            aligned = True
            for signal in required_signals:
                g = signal_groups[signal]
                ds = g["ds"].to_numpy()
                if len(ds) != len(reference_ds) or not np.array_equal(ds, reference_ds):
                    aligned = False
                    break
            if not aligned:
                continue
            signals_csv = ",".join(required_signals)

        series_id = str(target_rows["unique_id"].iloc[0])
        candidates.append(
            {
                "data": str(Path(data_path)),
                "series_id": series_id,
                "mode": mode,
                "signals_csv": signals_csv,
                "n_samples": int(len(target_rows)),
                "scenario": str(scenario),
                "target_signal": target_signal,
            }
        )

    if not candidates:
        raise ValueError(
            f"No eligible traces found for target_signal={target_signal!r}; "
            f"minimum required length is {min_n}"
        )

    result = (
        pd.DataFrame(candidates)
        .sort_values(["n_samples", "series_id"], ascending=[False, True])
        .head(int(n_traces))
        .reset_index(drop=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
