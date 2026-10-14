from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import rankdata


def _odd_window(window: int) -> int:
    window = max(3, int(window))
    return window if window % 2 == 1 else window + 1


def _aligned_signals(
    frame: pd.DataFrame,
    series_id: str,
    signals: tuple[str, ...],
) -> tuple[str, list[str], np.ndarray, np.ndarray]:
    anchor = frame.loc[frame["unique_id"].astype(str).eq(str(series_id))].copy()
    anchor = anchor.sort_values("ds").reset_index(drop=True)
    if anchor.empty:
        raise ValueError(f"series {series_id!r} not found")

    target_signal = str(anchor.iloc[0].get("signal", series_id))
    target = anchor["y"].to_numpy(float)
    anchor_ds = anchor["ds"].to_numpy()

    if not signals:
        return target_signal, [target_signal], target[:, None], anchor_ds

    required = {"scenario", "signal"}
    if not required.issubset(frame.columns):
        raise ValueError("multivariate precursor analysis requires scenario and signal columns")

    scenario = anchor.iloc[0]["scenario"]
    scope = frame.loc[frame["scenario"].eq(scenario)].copy()
    if "dataset" in frame.columns:
        scope = scope.loc[scope["dataset"].eq(anchor.iloc[0]["dataset"])]

    selected_signals = list(dict.fromkeys((target_signal, *signals)))
    arrays: list[np.ndarray] = []
    for signal in selected_signals:
        group = scope.loc[scope["signal"].astype(str).eq(str(signal))].copy()
        group = group.sort_values("ds").reset_index(drop=True)
        if len(group) != len(anchor) or not np.array_equal(group["ds"].to_numpy(), anchor_ds):
            raise ValueError(f"signal {signal!r} is missing or not exactly aligned with target")
        arrays.append(group["y"].to_numpy(float))

    return target_signal, selected_signals, np.column_stack(arrays), anchor_ds


def _event_locations(
    target: np.ndarray,
    *,
    prominence_fraction: float,
    prominence_window: int,
    event_distance: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    q01 = float(np.quantile(target, 0.01))
    q99 = float(np.quantile(target, 0.99))
    threshold = float(prominence_fraction) * (q99 - q01)
    if threshold < 0:
        raise ValueError("prominence threshold must be non-negative")

    kwargs = {
        "prominence": threshold,
        "wlen": _odd_window(prominence_window),
        "distance": max(1, int(event_distance)),
    }
    peaks, _ = find_peaks(target, **kwargs)
    troughs, _ = find_peaks(-target, **kwargs)
    return peaks.astype(int), troughs.astype(int), threshold


def _window_features(window: np.ndarray) -> np.ndarray:
    """Compact deterministic features from a [lookback, channels] history window."""
    if window.ndim != 2:
        raise ValueError("window must be [lookback, channels]")
    n = window.shape[0]
    x = np.linspace(-1.0, 1.0, n, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))

    first = window[0]
    last = window[-1]
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    minimum = window.min(axis=0)
    maximum = window.max(axis=0)
    delta = last - first
    centered = window - mean
    slope = (x_centered[:, None] * centered).sum(axis=0) / max(denom, 1e-12)

    return np.concatenate([last, mean, std, minimum, maximum, delta, slope]).astype(float)


def _future_event_labels(
    anchors: np.ndarray,
    event_locations: np.ndarray,
    horizon: int,
    tolerance: int,
) -> np.ndarray:
    if len(event_locations) == 0:
        return np.zeros(len(anchors), dtype=int)
    future = anchors + int(horizon)
    labels = np.zeros(len(anchors), dtype=int)
    events = np.asarray(event_locations, dtype=int)
    for i, index in enumerate(future):
        pos = np.searchsorted(events, index)
        candidates = []
        if pos < len(events):
            candidates.append(abs(int(events[pos]) - int(index)))
        if pos > 0:
            candidates.append(abs(int(events[pos - 1]) - int(index)))
        labels[i] = int(bool(candidates) and min(candidates) <= tolerance)
    return labels


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _point_biserial(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    if len(np.unique(labels)) < 2:
        return np.full(features.shape[1], np.nan)
    y_std = float(labels.std())
    out = np.full(features.shape[1], np.nan)
    for j in range(features.shape[1]):
        x = features[:, j]
        if not np.isfinite(x).all() or float(x.std()) == 0.0 or y_std == 0.0:
            continue
        out[j] = float(np.corrcoef(x, labels)[0, 1])
    return out


def _nearest_centroid_eval(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, float]:
    x_fit = features[fit_mask]
    y_fit = labels[fit_mask]
    x_eval = features[eval_mask]
    y_eval = labels[eval_mask]

    result = {
        "classifier_auc": float("nan"),
        "classifier_balanced_accuracy": float("nan"),
        "n_fit_positive": float((y_fit == 1).sum()),
        "n_fit_negative": float((y_fit == 0).sum()),
        "n_eval_positive": float((y_eval == 1).sum()),
        "n_eval_negative": float((y_eval == 0).sum()),
    }
    if len(x_fit) == 0 or len(x_eval) == 0 or len(np.unique(y_fit)) < 2 or len(np.unique(y_eval)) < 2:
        return result

    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z_fit = (x_fit - mean) / scale
    z_eval = (x_eval - mean) / scale

    positive_centroid = z_fit[y_fit == 1].mean(axis=0)
    negative_centroid = z_fit[y_fit == 0].mean(axis=0)
    d_pos = ((z_eval - positive_centroid) ** 2).sum(axis=1)
    d_neg = ((z_eval - negative_centroid) ** 2).sum(axis=1)
    scores = d_neg - d_pos
    predictions = (scores >= 0.0).astype(int)

    tpr = float((predictions[y_eval == 1] == 1).mean())
    tnr = float((predictions[y_eval == 0] == 0).mean())
    result["classifier_auc"] = _roc_auc(y_eval, scores)
    result["classifier_balanced_accuracy"] = 0.5 * (tpr + tnr)
    return result


def analyze_event_precursors(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    *,
    signals: tuple[str, ...] = (),
    input_size: int = 336,
    train_fraction: float = 0.7,
    analysis_val_fraction: float = 0.3,
    prominence_fraction: float = 0.10,
    prominence_window: int = 31,
    event_distance: int = 3,
    event_tolerance: int = 3,
) -> pd.DataFrame:
    """Measure event-specific precursor structure without touching the test split.

    The analysis uses only the forecasting experiment's training prefix. For each
    horizon, a history window ending at t is labeled positive when a prominence-
    qualified event occurs at t+h within +/- event_tolerance steps. A compact set
    of deterministic history features is evaluated in two ways: (1) the largest
    absolute point-biserial correlation with the event label and (2) a simple
    nearest-centroid classifier fit on the earlier part of the training prefix
    and evaluated on its later part. The classifier is intentionally lightweight;
    it is a diagnostic of precursor separability, not a competing forecaster.
    """
    if input_size < 3:
        raise ValueError("input_size must be at least 3")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < analysis_val_fraction < 1:
        raise ValueError("analysis_val_fraction must be in (0, 1)")

    target_signal, signal_names, values, _ = _aligned_signals(frame, series_id, signals)
    train_n = int(np.floor(len(values) * train_fraction))
    train_values = values[:train_n]
    target = train_values[:, 0]

    peaks, troughs, threshold = _event_locations(
        target,
        prominence_fraction=prominence_fraction,
        prominence_window=prominence_window,
        event_distance=event_distance,
    )
    event_sets = {
        "any": np.unique(np.concatenate([peaks, troughs])),
        "peak": peaks,
        "trough": troughs,
    }

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        last_anchor = train_n - horizon - 1
        first_anchor = input_size - 1
        if last_anchor < first_anchor:
            raise ValueError(f"training prefix is too short for input_size={input_size} and horizon={horizon}")
        anchors = np.arange(first_anchor, last_anchor + 1, dtype=int)
        features = np.vstack(
            [_window_features(train_values[t - input_size + 1 : t + 1]) for t in anchors]
        )

        split = int(np.floor(len(anchors) * (1.0 - analysis_val_fraction)))
        split = min(max(split, 1), len(anchors) - 1)
        fit_mask = np.arange(len(anchors)) < split
        eval_mask = ~fit_mask

        for event_type, locations in event_sets.items():
            labels = _future_event_labels(anchors, locations, horizon, event_tolerance)
            point_biserial = _point_biserial(features, labels)
            finite = np.isfinite(point_biserial)
            if finite.any():
                finite_indices = np.flatnonzero(finite)
                best_index = int(finite_indices[np.argmax(np.abs(point_biserial[finite]))])
                best_pb = float(point_biserial[best_index])
                statistic_names = ("last", "mean", "std", "min", "max", "delta", "slope")
                n_channels = len(signal_names)
                statistic = statistic_names[best_index // n_channels]
                signal = signal_names[best_index % n_channels]
                best_feature = f"{signal}:{statistic}"
                max_abs_pb = abs(best_pb)
            else:
                best_pb = float("nan")
                best_feature = None
                max_abs_pb = float("nan")
            classifier = _nearest_centroid_eval(features, labels, fit_mask, eval_mask)
            rows.append(
                {
                    "series_id": series_id,
                    "target_signal": target_signal,
                    "signals": ",".join(signal_names),
                    "n_signals": len(signal_names),
                    "horizon": int(horizon),
                    "event_type": event_type,
                    "input_size": int(input_size),
                    "train_fraction": float(train_fraction),
                    "analysis_val_fraction": float(analysis_val_fraction),
                    "prominence_fraction": float(prominence_fraction),
                    "prominence_threshold": threshold,
                    "prominence_window": int(prominence_window),
                    "event_distance": int(event_distance),
                    "event_tolerance": int(event_tolerance),
                    "n_samples": int(len(labels)),
                    "n_positive": int(labels.sum()),
                    "n_negative": int((labels == 0).sum()),
                    "positive_fraction": float(labels.mean()),
                    "max_abs_point_biserial_r": max_abs_pb,
                    "best_point_biserial_r": best_pb,
                    "best_precursor_feature": best_feature,
                    **classifier,
                }
            )
    return pd.DataFrame(rows)


def write_event_precursor_analysis(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    output: Path,
    **kwargs,
) -> Path:
    result = analyze_event_precursors(frame, series_id, horizons, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
