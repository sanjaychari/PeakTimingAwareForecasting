from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze_correlations(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    *,
    signals: tuple[str, ...] = (),
    train_fraction: float = 0.7,
) -> pd.DataFrame:
    """Measure Pearson lead correlation at the paper forecast horizons.

    For the target signal this is the standard lag-h autocorrelation
    corr(y_t, y_{t+h}). For auxiliary channels it is the lead cross-correlation
    corr(x_t, y_{t+h}). Only the training prefix is used so the analysis does
    not inspect held-out targets.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")

    anchor = frame.loc[frame["unique_id"].astype(str).eq(str(series_id))].copy()
    anchor = anchor.sort_values("ds").reset_index(drop=True)
    if anchor.empty:
        raise ValueError(f"series {series_id!r} not found")

    train_n = int(np.floor(len(anchor) * train_fraction))
    if train_n < max(horizons) + 3:
        raise ValueError("training prefix is too short for the requested horizons")

    target = anchor["y"].to_numpy(float)[:train_n]
    target_signal = str(anchor.iloc[0].get("signal", series_id))

    selected: list[tuple[str, np.ndarray, bool]] = [(target_signal, target, True)]
    if signals:
        required = {"scenario", "signal"}
        if not required.issubset(frame.columns):
            raise ValueError("multivariate correlation analysis requires scenario and signal columns")
        scenario = anchor.iloc[0]["scenario"]
        scope = frame.loc[frame["scenario"].eq(scenario)].copy()
        if "dataset" in frame.columns:
            scope = scope.loc[scope["dataset"].eq(anchor.iloc[0]["dataset"])]
        anchor_ds = anchor["ds"].to_numpy()
        selected = []
        for signal in signals:
            group = scope.loc[scope["signal"].astype(str).eq(str(signal))].copy()
            group = group.sort_values("ds").reset_index(drop=True)
            if len(group) != len(anchor) or not np.array_equal(group["ds"].to_numpy(), anchor_ds):
                raise ValueError(f"signal {signal!r} is missing or not exactly aligned with target")
            selected.append(
                (str(signal), group["y"].to_numpy(float)[:train_n], str(signal) == target_signal)
            )
        if target_signal not in {s for s, _, _ in selected}:
            selected.insert(0, (target_signal, target, True))

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        future_target = target[horizon:]
        for signal, values, is_target in selected:
            current = values[:-horizon]
            rows.append(
                {
                    "series_id": series_id,
                    "target_signal": target_signal,
                    "signal": signal,
                    "horizon": int(horizon),
                    "correlation_type": "autocorrelation" if is_target else "cross_correlation",
                    "pearson_r": _pearson(current, future_target),
                    "n_pairs": int(len(current)),
                    "train_fraction": float(train_fraction),
                }
            )
    return pd.DataFrame(rows)


def write_correlation_analysis(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    output: Path,
    **kwargs,
) -> Path:
    result = analyze_correlations(frame, series_id, horizons, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
