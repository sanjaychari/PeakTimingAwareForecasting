from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import infer_frequency, load_prepared
from .evaluation import EventConfig, _event_indices, _match_events, evaluate_forecast


@dataclass(frozen=True)
class BaselineConfig:
    input_size: int = 336
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    prominence_fraction: float = 0.10
    prominence_window: int = 31
    event_distance: int = 3
    event_tolerance: int = 3
    random_trials: int = 1000
    random_seed: int = 1


def _split_lengths(n: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    train_n = int(np.floor(n * train_fraction))
    val_n = int(np.floor(n * val_fraction))
    return train_n, val_n, n - train_n - val_n


def _event_config(y_train: np.ndarray, config: BaselineConfig) -> EventConfig:
    q01 = float(np.quantile(y_train, 0.01))
    q99 = float(np.quantile(y_train, 0.99))
    threshold = config.prominence_fraction * (q99 - q01)
    if threshold <= 0:
        raise ValueError("Training target has no positive Q0.99-Q0.01 range")
    return EventConfig(
        peak_prominence=threshold,
        trough_prominence=threshold,
        prominence_window=config.prominence_window,
        distance=config.event_distance,
        tolerance=config.event_tolerance,
    )


def _f1_from_indices(truth: np.ndarray, pred: np.ndarray, tolerance: int) -> float:
    matches = _match_events(np.asarray(truth, dtype=int), np.asarray(pred, dtype=int), tolerance)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(truth) - tp
    if len(truth) == 0 and len(pred) == 0:
        return float("nan")
    if len(truth) == 0 or len(pred) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _macro_event_f1(
    true_peaks: np.ndarray,
    true_troughs: np.ndarray,
    pred_peaks: np.ndarray,
    pred_troughs: np.ndarray,
    tolerance: int,
) -> float:
    values = [
        _f1_from_indices(true_peaks, pred_peaks, tolerance),
        _f1_from_indices(true_troughs, pred_troughs, tolerance),
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _sample_positions(
    rng: np.random.Generator,
    n_events: int,
    horizon: int,
    distance: int,
) -> np.ndarray:
    """Sample random event positions while respecting the event-distance rule."""
    if n_events <= 0:
        return np.empty(0, dtype=int)
    available = np.arange(horizon, dtype=int)
    selected: list[int] = []
    for _ in range(min(n_events, horizon)):
        valid = available
        if selected:
            valid = valid[
                np.all(
                    np.abs(valid[:, None] - np.asarray(selected, dtype=int)[None, :])
                    >= max(1, int(distance)),
                    axis=1,
                )
            ]
        if len(valid) == 0:
            break
        choice = int(rng.choice(valid))
        selected.append(choice)
    return np.asarray(sorted(selected), dtype=int)


def evaluate_trivial_baselines(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    *,
    config: BaselineConfig = BaselineConfig(),
) -> pd.DataFrame:
    series = frame.loc[
        frame["unique_id"].astype(str).eq(str(series_id)), ["unique_id", "ds", "y"]
    ].copy()
    if series.empty:
        raise ValueError(f"Series {series_id!r} not found")
    series = series.sort_values("ds").reset_index(drop=True)

    y_all = series["y"].to_numpy(float)
    train_n, val_n, _ = _split_lengths(len(series), config.train_fraction, config.val_fraction)
    y_train = y_all[:train_n]
    event_config = _event_config(y_train, config)
    train_peaks, _ = _event_indices(y_train, event_config, +1)
    train_troughs, _ = _event_indices(y_train, event_config, -1)
    peak_rate = len(train_peaks) / max(len(y_train), 1)
    trough_rate = len(train_troughs) / max(len(y_train), 1)
    test_start = train_n + val_n

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        starts = list(range(test_start, len(series) - horizon + 1, horizon))
        if not starts:
            continue

        persistence_scores = []
        replay_scores = []
        true_peak_count = 0
        true_trough_count = 0
        windows_with_event = 0
        truth_by_window = []

        for start in starts:
            y = y_all[start : start + horizon]
            true_peaks, _ = _event_indices(y, event_config, +1)
            true_troughs, _ = _event_indices(y, event_config, -1)
            truth_by_window.append((true_peaks, true_troughs))
            true_peak_count += len(true_peaks)
            true_trough_count += len(true_troughs)
            windows_with_event += int(len(true_peaks) + len(true_troughs) > 0)

            y_persistence = np.full(horizon, y_all[start - 1], dtype=float)
            persistence_scores.append(
                evaluate_forecast(y, y_persistence, event_config)["event_macro_f1"]
            )

            history_start = max(0, start - horizon)
            replay = y_all[history_start:start]
            if len(replay) < horizon:
                replay = np.pad(replay, (horizon - len(replay), 0), mode="edge")
            replay_scores.append(
                evaluate_forecast(y, replay[-horizon:], event_config)["event_macro_f1"]
            )

        def add_deterministic(name: str, scores: list[float]) -> None:
            arr = np.asarray(scores, dtype=float)
            rows.append(
                {
                    "series_id": series_id,
                    "horizon": int(horizon),
                    "baseline": name,
                    "n_test_windows": len(starts),
                    "n_windows_with_true_event": windows_with_event,
                    "n_true_peaks": true_peak_count,
                    "n_true_troughs": true_trough_count,
                    "train_peak_rate_per_step": peak_rate,
                    "train_trough_rate_per_step": trough_rate,
                    "event_macro_f1_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
                    "event_macro_f1_std": float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan"),
                    "event_macro_f1_ci_low": float("nan"),
                    "event_macro_f1_ci_high": float("nan"),
                    "random_trials": 0,
                }
            )

        add_deterministic("persistence", persistence_scores)
        add_deterministic("recent_window_replay", replay_scores)

        rng = np.random.default_rng(config.random_seed + int(horizon))
        trial_means = []
        for _ in range(config.random_trials):
            window_scores = []
            for true_peaks, true_troughs in truth_by_window:
                n_peak = int(rng.poisson(peak_rate * horizon))
                n_trough = int(rng.poisson(trough_rate * horizon))
                pred_peaks = _sample_positions(rng, n_peak, horizon, config.event_distance)
                pred_troughs = _sample_positions(rng, n_trough, horizon, config.event_distance)
                window_scores.append(
                    _macro_event_f1(
                        true_peaks,
                        true_troughs,
                        pred_peaks,
                        pred_troughs,
                        config.event_tolerance,
                    )
                )
            arr = np.asarray(window_scores, dtype=float)
            trial_means.append(float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"))

        trial_arr = np.asarray(trial_means, dtype=float)
        finite = trial_arr[np.isfinite(trial_arr)]
        rows.append(
            {
                "series_id": series_id,
                "horizon": int(horizon),
                "baseline": "event_rate_random",
                "n_test_windows": len(starts),
                "n_windows_with_true_event": windows_with_event,
                "n_true_peaks": true_peak_count,
                "n_true_troughs": true_trough_count,
                "train_peak_rate_per_step": peak_rate,
                "train_trough_rate_per_step": trough_rate,
                "event_macro_f1_mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "event_macro_f1_std": float(np.std(finite, ddof=0)) if len(finite) else float("nan"),
                "event_macro_f1_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
                "event_macro_f1_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
                "random_trials": int(config.random_trials),
            }
        )

    return pd.DataFrame(rows)


def write_trivial_baselines(
    frame: pd.DataFrame,
    series_id: str,
    horizons: tuple[int, ...],
    output: Path,
    **kwargs,
) -> Path:
    result = evaluate_trivial_baselines(
        frame,
        series_id,
        horizons,
        config=BaselineConfig(**kwargs),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
