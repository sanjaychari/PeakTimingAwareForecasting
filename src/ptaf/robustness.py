from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import EventConfig, evaluate_forecast


def _focus_calibration(metadata: dict) -> tuple[str | None, float]:
    """Return the focus signal and training Q99-Q01 central range.

    Experiment metadata evolved over time.  Recover the calibration from,
    in order of preference:

      1. the stored central range,
      2. stored training Q01/Q99 values,
      3. the stored raw prominence threshold and prominence fraction.

    The third case is exact because experiments define

        prominence_threshold_raw =
            prominence_fraction * (Q99_train - Q01_train).
    """

    def central_range(record: dict) -> float:
        value = record.get("prominence_train_central_range")
        if value is not None:
            return float(value)

        q01 = record.get("prominence_train_q01")
        q99 = record.get("prominence_train_q99")
        if q01 is not None and q99 is not None:
            return float(q99) - float(q01)

        q01 = record.get("q01_train")
        q99 = record.get("q99_train")
        if q01 is not None and q99 is not None:
            return float(q99) - float(q01)

        threshold = record.get("peak_prominence_threshold_raw")
        if threshold is None:
            threshold = record.get("trough_prominence_threshold_raw")

        if threshold is not None:
            fraction = record.get(
                "prominence_fraction",
                metadata.get("prominence_fraction", 0.10),
            )
            fraction = float(fraction)
            if fraction <= 0:
                raise ValueError(
                    f"Invalid prominence_fraction={fraction!r}"
                )
            return float(threshold) / fraction

        raise KeyError(
            "config does not contain enough information to recover the "
            "training prominence calibration"
        )

    if metadata.get("multivariate"):
        focus_signal = metadata.get("focus_signal")
        for channel in metadata.get("channels", []):
            if channel.get("signal") == focus_signal:
                return focus_signal, central_range(channel)

        raise ValueError(
            f"Could not find focus-signal calibration for "
            f"{focus_signal!r} in multivariate config"
        )

    return None, central_range(metadata)


def _global_f1(tp: float, fp: float, fn: float) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if not np.isfinite(precision) or not np.isfinite(recall):
        return 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def analyze_run_sensitivity(
    run_dir: Path,
    *,
    prominence_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    event_tolerances: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    config_path = run_dir / "config.json"
    forecast_path = run_dir / "forecast.csv"
    if not config_path.exists() or not forecast_path.exists():
        raise FileNotFoundError(f"Expected config.json and forecast.csv under {run_dir}")

    metadata = json.loads(config_path.read_text())
    forecast = pd.read_csv(forecast_path)
    focus_signal, central_range = _focus_calibration(metadata)
    if focus_signal is not None:
        forecast = forecast.loc[forecast["signal"].astype(str).eq(str(focus_signal))].copy()
    if forecast.empty:
        raise ValueError(f"No focus-signal forecasts in {run_dir}")

    prominence_window = int(metadata.get("prominence_window", 31))
    event_distance = int(metadata.get("event_distance", 3))
    rows: list[dict[str, object]] = []

    for prominence_fraction in prominence_fractions:
        threshold = float(prominence_fraction) * central_range
        for tolerance in event_tolerances:
            event_config = EventConfig(
                peak_prominence=threshold,
                trough_prominence=threshold,
                prominence_window=prominence_window,
                distance=event_distance,
                tolerance=int(tolerance),
            )
            metrics = []
            for _, group in forecast.groupby("window", sort=True):
                group = group.sort_values("ds")
                metrics.append(
                    evaluate_forecast(
                        group["y"].to_numpy(float),
                        group["y_hat"].to_numpy(float),
                        event_config,
                    )
                )
            metric_frame = pd.DataFrame(metrics)

            peak_tp = float(metric_frame["peak_tp"].sum())
            peak_fp = float(metric_frame["peak_fp"].sum())
            peak_fn = float(metric_frame["peak_fn"].sum())
            trough_tp = float(metric_frame["trough_tp"].sum())
            trough_fp = float(metric_frame["trough_fp"].sum())
            trough_fn = float(metric_frame["trough_fn"].sum())
            global_peak_f1 = _global_f1(peak_tp, peak_fp, peak_fn)
            global_trough_f1 = _global_f1(trough_tp, trough_fp, trough_fn)
            finite_global = [v for v in (global_peak_f1, global_trough_f1) if np.isfinite(v)]

            rows.append(
                {
                    "run_dir": str(run_dir),
                    "series_id": metadata.get("series_id"),
                    "model": metadata.get("model"),
                    "loss": metadata.get("loss"),
                    "horizon": metadata.get("horizon"),
                    "seed": metadata.get("seed"),
                    "multivariate": bool(metadata.get("multivariate", False)),
                    "focus_signal": focus_signal,
                    "prominence_fraction": float(prominence_fraction),
                    "prominence_threshold": threshold,
                    "event_tolerance": int(tolerance),
                    "n_windows": int(len(metric_frame)),
                    "window_event_macro_f1_mean": float(metric_frame["event_macro_f1"].mean()),
                    "window_event_macro_f1_median": float(metric_frame["event_macro_f1"].median()),
                    "global_peak_f1": global_peak_f1,
                    "global_trough_f1": global_trough_f1,
                    "global_event_macro_f1": float(np.mean(finite_global)) if finite_global else float("nan"),
                    "peak_tp": peak_tp,
                    "peak_fp": peak_fp,
                    "peak_fn": peak_fn,
                    "trough_tp": trough_tp,
                    "trough_fp": trough_fp,
                    "trough_fn": trough_fn,
                }
            )
    return pd.DataFrame(rows)


def analyze_results_sensitivity(
    results_root: Path,
    *,
    prominence_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    event_tolerances: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Evaluate sensitivity over exactly the configurations used in the paper."""

    paper_series = {
        "telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes",
        "mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct",
        "esnet::r-nova-fd-01::ifHCOutOctets",
    }
    paper_losses = {"mse", "ep", "mep"}
    paper_horizons = {96, 192, 336, 720}

    # Models evaluated in each input setting in the paper.
    univariate_models = {
        "dlinear",
        "patchtst",
        "itransformer",
        "timemixer",
    }
    multivariate_models = {
        "tsmixer",
        "itransformer",
        "timemixer",
    }

    frames = []
    selected_runs = []

    for config_path in sorted(results_root.rglob("config.json")):
        run_dir = config_path.parent
        forecast_path = run_dir / "forecast.csv"
        if not forecast_path.exists():
            continue

        metadata = json.loads(config_path.read_text())

        if metadata.get("series_id") not in paper_series:
            continue
        if metadata.get("loss") not in paper_losses:
            continue
        if metadata.get("horizon") not in paper_horizons:
            continue

        multivariate = bool(metadata.get("multivariate", False))
        model = metadata.get("model")

        # MIT Supercloud is evaluated only in the univariate setting.
        if (
            metadata.get("series_id")
            == "mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct"
            and multivariate
        ):
            continue

        if multivariate:
            if model not in multivariate_models:
                continue
        else:
            if model not in univariate_models:
                continue

        selected_runs.append(run_dir)

        frames.append(
            analyze_run_sensitivity(
                run_dir,
                prominence_fractions=prominence_fractions,
                event_tolerances=event_tolerances,
            )
        )

    if not frames:
        raise ValueError(
            f"No paper experiment runs found under {results_root}"
        )

    if len(selected_runs) != 216:
        raise ValueError(
            "Expected exactly 216 paper configurations, "
            f"but selected {len(selected_runs)}. "
            "Check the results directory before using the sensitivity output."
        )

    return pd.concat(frames, ignore_index=True)


def write_results_sensitivity(
    results_root: Path,
    output: Path,
    **kwargs,
) -> Path:
    result = analyze_results_sensitivity(results_root, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
