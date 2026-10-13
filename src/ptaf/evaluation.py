from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class EventConfig:
    peak_prominence: float
    trough_prominence: float
    prominence_window: int = 31
    distance: int = 3
    tolerance: int = 3


def _odd_window(window: int) -> int:
    window = max(3, int(window))
    return window if window % 2 == 1 else window + 1


def _event_indices(values: np.ndarray, config: EventConfig, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Return prominence-qualified peak or trough locations.

    Peaks are found directly on ``values``; troughs are peaks of ``-values``.
    ``wlen`` bounds the prominence calculation to a local neighborhood while
    preserving scipy.signal.find_peaks' standard prominence definition.
    """
    signal = values if sign > 0 else -values
    threshold = config.peak_prominence if sign > 0 else config.trough_prominence
    peaks, props = find_peaks(
        signal,
        prominence=max(float(threshold), 0.0),
        wlen=_odd_window(config.prominence_window),
        distance=max(1, int(config.distance)),
    )
    prominences = np.asarray(props.get("prominences", np.empty(0)), dtype=float)
    return peaks.astype(int), prominences


def _match_events(target: np.ndarray, pred: np.ndarray, tolerance: int) -> list[tuple[int, int]]:
    candidates = []
    for i, t in enumerate(target):
        for j, p in enumerate(pred):
            d = abs(int(t) - int(p))
            if d <= tolerance:
                candidates.append((d, i, j))
    candidates.sort()
    used_t: set[int] = set()
    used_p: set[int] = set()
    matches = []
    for _, i, j in candidates:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matches.append((int(target[i]), int(pred[j])))
    return matches


def _event_metrics(y: np.ndarray, y_hat: np.ndarray, config: EventConfig, sign: int) -> dict[str, float]:
    truth, truth_prominence = _event_indices(y, config, sign)
    pred, pred_prominence = _event_indices(y_hat, config, sign)
    matches = _match_events(truth, pred, config.tolerance)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(truth) - tp
    if len(truth) == 0 and len(pred) == 0:
        precision = recall = f1 = float("nan")
    elif len(truth) == 0:
        precision, recall, f1 = 0.0, float("nan"), 0.0
    elif len(pred) == 0:
        precision, recall, f1 = float("nan"), 0.0, 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if matches:
        timing = float(np.mean([abs(t - p) for t, p in matches]))
        magnitude = float(np.mean([abs(y[t] - y_hat[p]) for t, p in matches]))
    else:
        timing = float("nan")
        magnitude = float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "timing_mae_steps": timing,
        "magnitude_mae": magnitude,
        "n_target_events": float(len(truth)),
        "n_pred_events": float(len(pred)),
        "mean_target_prominence": float(np.mean(truth_prominence)) if len(truth_prominence) else float("nan"),
        "mean_pred_prominence": float(np.mean(pred_prominence)) if len(pred_prominence) else float("nan"),
    }


def evaluate_forecast(y: np.ndarray, y_hat: np.ndarray, config: EventConfig) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape")
    out = {
        "mse": float(np.mean((y - y_hat) ** 2)),
        "mae": float(np.mean(np.abs(y - y_hat))),
    }
    pos = _event_metrics(y, y_hat, config, +1)
    neg = _event_metrics(y, y_hat, config, -1)
    for prefix, values in (("peak", pos), ("trough", neg)):
        out.update({f"{prefix}_{k}": v for k, v in values.items()})
    f1_values = [v for v in (pos["f1"], neg["f1"]) if np.isfinite(v)]
    out["event_macro_f1"] = float(np.mean(f1_values)) if f1_values else float("nan")
    timing_values = [v for v in (pos["timing_mae_steps"], neg["timing_mae_steps"]) if np.isfinite(v)]
    magnitude_values = [v for v in (pos["magnitude_mae"], neg["magnitude_mae"]) if np.isfinite(v)]
    out["event_timing_mae_steps"] = float(np.mean(timing_values)) if timing_values else float("nan")
    out["event_magnitude_mae"] = float(np.mean(magnitude_values)) if magnitude_values else float("nan")
    return out


def aggregate_window_metrics(metrics: list[dict[str, float]]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    frame = pd.DataFrame(metrics)
    records = []
    metadata_columns = {"window", "start_ds", "signal", "unique_id", "source_unique_id"}
    for column in frame.columns:
        if column in metadata_columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        records.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "n_valid": int(values.notna().sum()),
            }
        )
    return pd.DataFrame.from_records(records)
