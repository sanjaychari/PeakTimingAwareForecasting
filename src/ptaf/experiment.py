from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import infer_frequency, load_prepared
from .evaluation import EventConfig, aggregate_window_metrics, evaluate_forecast
from .losses import LossConfig, build_loss
from .models import ModelConfig, build_model
from .paths import results_root


@dataclass(frozen=True)
class ExperimentConfig:
    data: Path
    series_id: str
    model: str
    loss: str
    horizon: int = 96
    input_size: int = 336
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    max_steps: int = 1000
    tuning_max_steps: int = 100
    learning_rate: float = 1e-3
    seed: int = 1
    prominence_fraction: float = 0.10
    prominence_window: int = 31
    event_distance: int = 3
    event_tolerance: int = 3
    ep_quantile: float = 0.90
    mep_low_quantile: float = 0.10
    mep_high_quantile: float = 0.90
    ep_under_weight: float = 2.0
    ep_over_weight: float = 1.0
    ep_quantile_grid: tuple[float, ...] = (0.90, 0.95)
    mep_low_quantile_grid: tuple[float, ...] = (0.05, 0.10)
    mep_high_quantile_grid: tuple[float, ...] = (0.90, 0.95)
    ep_under_weight_grid: tuple[float, ...] = (1.0, 2.0)
    ep_over_weight_grid: tuple[float, ...] = (1.0, 2.0)
    tuning_metric: str = "event_macro_f1"
    multivariate_signals: tuple[str, ...] = ()


def _split_lengths(n: int, train_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    train_n = int(np.floor(n * train_fraction))
    val_n = int(np.floor(n * val_fraction))
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def _scale_from_train(y: np.ndarray) -> float:
    q95 = float(np.quantile(np.abs(y), 0.95))
    std = float(np.std(y))
    return max(q95, std, 1e-8)


def _identify_prediction_column(pred: pd.DataFrame, alias: str) -> str:
    if alias in pred.columns:
        return alias
    candidate_cols = [c for c in pred.columns if c not in {"unique_id", "ds", "cutoff"}]
    if len(candidate_cols) != 1:
        raise RuntimeError(f"Could not identify forecast column in {list(pred.columns)}")
    return candidate_cols[0]


def _calibration(y_train_raw: np.ndarray, config: ExperimentConfig) -> dict[str, float]:
    value_scale = _scale_from_train(y_train_raw)
    q01_train = float(np.quantile(y_train_raw, 0.01))
    q99_train = float(np.quantile(y_train_raw, 0.99))
    train_central_range = q99_train - q01_train
    if train_central_range <= 0:
        raise ValueError("Training target has no positive Q0.99-Q0.01 range for prominence calibration")
    prominence_threshold_raw = config.prominence_fraction * train_central_range
    return {
        "value_scale": value_scale,
        "q01_train": q01_train,
        "q99_train": q99_train,
        "train_central_range": train_central_range,
        "peak_prominence_threshold_raw": prominence_threshold_raw,
        "trough_prominence_threshold_raw": prominence_threshold_raw,
        "peak_prominence_threshold_scaled": prominence_threshold_raw / value_scale,
        "trough_prominence_threshold_scaled": prominence_threshold_raw / value_scale,
    }


def _channel_values(values: list[float]) -> float | tuple[float, ...]:
    return values[0] if len(values) == 1 else tuple(values)


def _build_loss(
    config: ExperimentConfig,
    train_raws: list[np.ndarray],
    calibrations: list[dict[str, float]],
):
    ep_thresholds = [
        float(np.quantile(y, config.ep_quantile) / calibration["value_scale"])
        for y, calibration in zip(train_raws, calibrations)
    ]
    mep_low_thresholds = [
        float(np.quantile(y, config.mep_low_quantile) / calibration["value_scale"])
        for y, calibration in zip(train_raws, calibrations)
    ]
    mep_high_thresholds = [
        float(np.quantile(y, config.mep_high_quantile) / calibration["value_scale"])
        for y, calibration in zip(train_raws, calibrations)
    ]
    return build_loss(
        LossConfig(
            name=config.loss,
            ep_threshold=_channel_values(ep_thresholds),
            mep_low_threshold=_channel_values(mep_low_thresholds),
            mep_high_threshold=_channel_values(mep_high_thresholds),
            ep_under_weight=config.ep_under_weight,
            ep_over_weight=config.ep_over_weight,
        )
    )


def _build_nf(config, loss, freq_alias: str, n_series: int):
    from neuralforecast import NeuralForecast

    model = build_model(
        ModelConfig(
            name=config.model,
            horizon=config.horizon,
            input_size=config.input_size,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            seed=config.seed,
            n_series=n_series,
        ),
        loss,
    )
    return NeuralForecast(models=[model], freq=freq_alias), model.alias


def _event_config(config: ExperimentConfig, calibration: dict[str, float]) -> EventConfig:
    return EventConfig(
        peak_prominence=calibration["peak_prominence_threshold_raw"],
        trough_prominence=calibration["trough_prominence_threshold_raw"],
        prominence_window=config.prominence_window,
        distance=config.event_distance,
        tolerance=config.event_tolerance,
    )


def _loss_candidates(config: ExperimentConfig) -> list[ExperimentConfig]:
    if config.loss == "mse":
        return [config]
    if config.loss == "ep":
        return [
            replace(config, ep_quantile=q, ep_under_weight=fu, ep_over_weight=fo)
            for q, fu, fo in product(
                config.ep_quantile_grid,
                config.ep_under_weight_grid,
                config.ep_over_weight_grid,
            )
        ]
    if config.loss == "mep":
        candidates = []
        for low_q, high_q, fu, fo in product(
            config.mep_low_quantile_grid,
            config.mep_high_quantile_grid,
            config.ep_under_weight_grid,
            config.ep_over_weight_grid,
        ):
            if not 0 < low_q < high_q < 1:
                continue
            candidates.append(
                replace(
                    config,
                    mep_low_quantile=low_q,
                    mep_high_quantile=high_q,
                    ep_under_weight=fu,
                    ep_over_weight=fo,
                )
            )
        if not candidates:
            raise ValueError("MEP grid contains no threshold pair satisfying 0 < low < high < 1")
        return candidates
    raise ValueError("loss must be one of: mse, ep, mep")


def _candidate_record(config: ExperimentConfig, metrics: list[dict[str, float]]) -> dict[str, float | str]:
    frame = pd.DataFrame(metrics)
    mean_mse = float(pd.to_numeric(frame["mse"], errors="coerce").mean())
    mean_f1 = float(pd.to_numeric(frame["event_macro_f1"], errors="coerce").mean())
    return {
        "loss": config.loss,
        "ep_quantile": config.ep_quantile,
        "mep_low_quantile": config.mep_low_quantile,
        "mep_high_quantile": config.mep_high_quantile,
        "under_weight": config.ep_under_weight,
        "over_weight": config.ep_over_weight,
        "validation_event_macro_f1": mean_f1,
        "validation_mse": mean_mse,
    }


def _is_better_candidate(record: dict, best: dict | None, tuning_metric: str) -> bool:
    if best is None:
        return True
    if tuning_metric == "mse":
        return record["validation_mse"] < best["validation_mse"]
    if tuning_metric != "event_macro_f1":
        raise ValueError("tuning_metric must be one of: event_macro_f1, mse")
    score = record["validation_event_macro_f1"]
    best_score = best["validation_event_macro_f1"]
    score = -np.inf if not np.isfinite(score) else score
    best_score = -np.inf if not np.isfinite(best_score) else best_score
    if score != best_score:
        return score > best_score
    return record["validation_mse"] < best["validation_mse"]


def _validate_split_lengths(n: int, config: ExperimentConfig) -> tuple[int, int, int]:
    train_n, val_n, test_n = _split_lengths(n, config.train_fraction, config.val_fraction)
    if train_n < config.input_size + config.horizon:
        raise ValueError("Training split is too short for input_size + horizon")
    if val_n < config.horizon:
        raise ValueError("Validation split must contain at least one forecast horizon")
    if test_n < config.horizon:
        raise ValueError("Test split must contain at least one forecast horizon")
    return train_n, val_n, test_n


def _fit_select_univariate(
    config: ExperimentConfig,
    series: pd.DataFrame,
    train_n: int,
    val_n: int,
    train_raw: np.ndarray,
    calibration: dict[str, float],
    freq_alias: str,
):
    trainval_end = train_n + val_n
    trainval = series.iloc[:trainval_end][["unique_id", "ds", "y"]].copy()
    candidates = _loss_candidates(config)
    records = []
    best_record = None
    best = None
    for candidate in candidates:
        fit_config = (
            candidate
            if config.loss == "mse"
            else replace(candidate, max_steps=config.tuning_max_steps)
        )
        loss = _build_loss(fit_config, [train_raw], [calibration])
        nf, alias = _build_nf(fit_config, loss, freq_alias, 1)
        nf.fit(df=trainval, val_size=val_n)
        metrics = []
        for start in range(train_n, trainval_end - config.horizon + 1, config.horizon):
            history = series.iloc[:start][["unique_id", "ds", "y"]].copy()
            pred = nf.predict(df=history)
            pred_col = _identify_prediction_column(pred, alias)
            y_hat = pred[pred_col].to_numpy(float)[: config.horizon] * calibration["value_scale"]
            y = series.iloc[start : start + config.horizon]["y_raw"].to_numpy(float)
            metrics.append(evaluate_forecast(y, y_hat, _event_config(config, calibration)))
        record = _candidate_record(candidate, metrics)
        records.append(record)
        if _is_better_candidate(record, best_record, config.tuning_metric):
            best_record = record
            best = (candidate, nf, alias)
    if best is None:
        raise RuntimeError("No loss candidate could be selected")
    selected_config, selected_nf, selected_alias = best
    if config.loss != "mse":
        final_loss = _build_loss(selected_config, [train_raw], [calibration])
        selected_nf, selected_alias = _build_nf(selected_config, final_loss, freq_alias, 1)
        selected_nf.fit(df=trainval, val_size=val_n)
    return selected_config, selected_nf, selected_alias, records


def _run_univariate(config: ExperimentConfig, output_dir: Path | None) -> Path:
    frame = load_prepared(config.data)
    series = frame.loc[
        frame["unique_id"].astype(str).eq(str(config.series_id)), ["unique_id", "ds", "y"]
    ].copy()
    if series.empty:
        choices = frame["unique_id"].drop_duplicates().astype(str).tolist()
        raise ValueError(f"Series {config.series_id!r} not found. Available examples: {choices[:10]}")
    series = series.sort_values("ds").reset_index(drop=True)
    freq = infer_frequency(series)
    train_n, val_n, test_n = _validate_split_lengths(len(series), config)

    train_y_raw = series.loc[: train_n - 1, "y"].to_numpy(float)
    calibration = _calibration(train_y_raw, config)
    value_scale = calibration["value_scale"]
    series["y_raw"] = series["y"]
    series["y"] = series["y"] / value_scale
    freq_alias = pd.tseries.frequencies.to_offset(freq).freqstr

    selected_config, nf, alias, tuning_records = _fit_select_univariate(
        config, series, train_n, val_n, train_y_raw, calibration, freq_alias
    )

    window_metrics = []
    forecast_rows = []
    trainval_end = train_n + val_n
    starts = range(trainval_end, len(series) - config.horizon + 1, config.horizon)
    event_config = _event_config(config, calibration)
    for window_index, start in enumerate(starts):
        history = series.iloc[:start][["unique_id", "ds", "y"]].copy()
        pred = nf.predict(df=history)
        pred_col = _identify_prediction_column(pred, alias)
        y_hat = pred[pred_col].to_numpy(float)[: config.horizon] * value_scale
        target = series.iloc[start : start + config.horizon]
        y = target["y_raw"].to_numpy(float)
        if len(y_hat) != len(y):
            raise RuntimeError("Forecast horizon length does not match target window")
        metrics = evaluate_forecast(y, y_hat, event_config)
        metrics["window"] = window_index
        metrics["start_ds"] = target["ds"].iloc[0]
        window_metrics.append(metrics)
        forecast_rows.append(pd.DataFrame({"window": window_index, "ds": target["ds"].to_numpy(), "y": y, "y_hat": y_hat}))

    if not window_metrics:
        raise ValueError("No complete test windows were available")

    run_name = (
        f"{config.series_id.replace('/', '_').replace(':', '_')}__{config.model}__"
        f"{config.loss}__h{config.horizon}__seed{config.seed}"
    )
    output_dir = output_dir or results_root() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(forecast_rows, ignore_index=True).to_csv(output_dir / "forecast.csv", index=False)
    pd.DataFrame(window_metrics).to_csv(output_dir / "window_metrics.csv", index=False)
    aggregate_window_metrics(window_metrics).to_csv(output_dir / "metrics.csv", index=False)
    if config.loss != "mse":
        pd.DataFrame(tuning_records).to_csv(output_dir / "loss_grid_search.csv", index=False)

    metadata = asdict(config)
    metadata["data"] = str(config.data)
    metadata["selected_loss_hyperparameters"] = {
        "ep_quantile": selected_config.ep_quantile if config.loss == "ep" else None,
        "mep_low_quantile": selected_config.mep_low_quantile if config.loss == "mep" else None,
        "mep_high_quantile": selected_config.mep_high_quantile if config.loss == "mep" else None,
        "under_weight": selected_config.ep_under_weight if config.loss != "mse" else None,
        "over_weight": selected_config.ep_over_weight if config.loss != "mse" else None,
        "selection_metric": config.tuning_metric if config.loss != "mse" else None,
    }
    metadata.update({
        "frequency": str(freq), "neuralforecast_frequency": freq_alias,
        "n_total": len(series), "n_train": train_n, "n_val": val_n, "n_test": test_n,
        "value_scale": value_scale,
        "peak_prominence_threshold_raw": calibration["peak_prominence_threshold_raw"],
        "trough_prominence_threshold_raw": calibration["trough_prominence_threshold_raw"],
        "prominence_train_q01": calibration["q01_train"],
        "prominence_train_q99": calibration["q99_train"],
        "prominence_train_central_range": calibration["train_central_range"],
        "n_test_windows": len(window_metrics), "multivariate": False,
    })
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return output_dir


def _select_multivariate_channels(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[list[dict[str, str]], list[pd.DataFrame]]:
    required = {"unique_id", "ds", "y", "signal", "scenario"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Full multivariate forecasting requires prepared data columns "
            f"{sorted(required)}; missing {sorted(missing)}"
        )

    anchor = frame.loc[frame["unique_id"].astype(str).eq(str(config.series_id))]
    if anchor.empty:
        choices = frame["unique_id"].drop_duplicates().astype(str).tolist()
        raise ValueError(f"Series {config.series_id!r} not found. Available examples: {choices[:10]}")
    anchor_row = anchor.iloc[0]
    scenario = anchor_row["scenario"]
    dataset = anchor_row["dataset"] if "dataset" in frame.columns else None

    signals = tuple(config.multivariate_signals)
    if len(signals) < 2:
        raise ValueError("Full multivariate forecasting requires at least two --multivariate-signals")
    if len(set(signals)) != len(signals):
        raise ValueError("--multivariate-signals must not contain duplicates")

    scope = frame.loc[frame["scenario"].eq(scenario)].copy()
    if dataset is not None:
        scope = scope.loc[scope["dataset"].eq(dataset)].copy()

    channel_info: list[dict[str, str]] = []
    channel_frames: list[pd.DataFrame] = []
    reference_ds: np.ndarray | None = None
    for index, signal in enumerate(signals):
        matches = scope.loc[scope["signal"].astype(str).eq(str(signal))].copy()
        ids = matches["unique_id"].drop_duplicates().astype(str).tolist()
        if len(ids) != 1:
            raise ValueError(
                f"Expected exactly one series for signal {signal!r} in scenario {scenario!r}; found {ids}"
            )
        source_id = ids[0]
        channel = matches.loc[matches["unique_id"].astype(str).eq(source_id), ["ds", "y"]].copy()
        channel = channel.sort_values("ds").reset_index(drop=True)
        ds = channel["ds"].to_numpy()
        if reference_ds is None:
            reference_ds = ds
        elif len(ds) != len(reference_ds) or not np.array_equal(ds, reference_ds):
            raise ValueError(f"Signal {signal!r} is not exactly timestamp-aligned with the other selected channels")
        nf_id = f"channel_{index:03d}"
        channel["unique_id"] = nf_id
        channel["signal"] = str(signal)
        channel["source_unique_id"] = source_id
        channel_info.append({"signal": str(signal), "source_unique_id": source_id, "neuralforecast_unique_id": nf_id})
        channel_frames.append(channel)
    return channel_info, channel_frames


def _fit_select_multivariate(
    config: ExperimentConfig,
    scaled_channels: list[pd.DataFrame],
    raw_by_id: dict[str, pd.DataFrame],
    channel_info: list[dict[str, str]],
    train_raws: list[np.ndarray],
    calibrations: list[dict[str, float]],
    train_n: int,
    val_n: int,
    freq_alias: str,
    focus_signal: str,
):
    trainval_end = train_n + val_n
    trainval = pd.concat(
        [channel.iloc[:trainval_end][["unique_id", "ds", "y"]] for channel in scaled_channels],
        ignore_index=True,
    )
    focus_index = next(i for i, info in enumerate(channel_info) if info["signal"] == focus_signal)
    focus_info = channel_info[focus_index]
    focus_calibration = calibrations[focus_index]
    focus_raw = raw_by_id[focus_info["neuralforecast_unique_id"]]

    records = []
    best_record = None
    best = None
    for candidate in _loss_candidates(config):
        fit_config = (
            candidate
            if config.loss == "mse"
            else replace(candidate, max_steps=config.tuning_max_steps)
        )
        loss = _build_loss(fit_config, train_raws, calibrations)
        nf, alias = _build_nf(fit_config, loss, freq_alias, len(scaled_channels))
        nf.fit(df=trainval, val_size=val_n)
        metrics = []
        for start in range(train_n, trainval_end - config.horizon + 1, config.horizon):
            history = pd.concat(
                [channel.iloc[:start][["unique_id", "ds", "y"]] for channel in scaled_channels],
                ignore_index=True,
            )
            pred = nf.predict(df=history)
            pred_col = _identify_prediction_column(pred, alias)
            pred_focus = pred.loc[pred["unique_id"].astype(str).eq(focus_info["neuralforecast_unique_id"])].sort_values("ds")
            y_hat = pred_focus[pred_col].to_numpy(float)[: config.horizon] * focus_calibration["value_scale"]
            y = focus_raw.iloc[start : start + config.horizon]["y"].to_numpy(float)
            metrics.append(evaluate_forecast(y, y_hat, _event_config(config, focus_calibration)))
        record = _candidate_record(candidate, metrics)
        records.append(record)
        if _is_better_candidate(record, best_record, config.tuning_metric):
            best_record = record
            best = (candidate, nf, alias)
    if best is None:
        raise RuntimeError("No loss candidate could be selected")
    selected_config, selected_nf, selected_alias = best
    if config.loss != "mse":
        final_loss = _build_loss(selected_config, train_raws, calibrations)
        selected_nf, selected_alias = _build_nf(
            selected_config, final_loss, freq_alias, len(scaled_channels)
        )
        selected_nf.fit(df=trainval, val_size=val_n)
    return selected_config, selected_nf, selected_alias, records


def _run_multivariate(config: ExperimentConfig, output_dir: Path | None) -> Path:
    frame = load_prepared(config.data)
    channel_info, channels = _select_multivariate_channels(frame, config)
    n = len(channels[0])
    freq = infer_frequency(channels[0])
    train_n, val_n, test_n = _validate_split_lengths(n, config)

    calibrations = []
    train_raws = []
    scaled_channels = []
    raw_by_id = {}
    for info, channel in zip(channel_info, channels):
        raw = channel["y"].to_numpy(float)
        train_raws.append(raw[:train_n])
        calibration = _calibration(raw[:train_n], config)
        calibrations.append(calibration)
        scaled = channel[["unique_id", "ds", "y"]].copy()
        scaled["y"] = scaled["y"] / calibration["value_scale"]
        scaled_channels.append(scaled)
        raw_by_id[info["neuralforecast_unique_id"]] = channel.copy()
        info.update({
            "value_scale": calibration["value_scale"],
            "peak_prominence_threshold_raw": calibration["peak_prominence_threshold_raw"],
            "trough_prominence_threshold_raw": calibration["trough_prominence_threshold_raw"],
            "prominence_train_q01": calibration["q01_train"],
            "prominence_train_q99": calibration["q99_train"],
            "prominence_train_central_range": calibration["train_central_range"],
        })

    anchor_signal = str(frame.loc[frame["unique_id"].astype(str).eq(str(config.series_id)), "signal"].iloc[0])
    if anchor_signal not in {info["signal"] for info in channel_info}:
        raise ValueError("The target/focus signal must be included in --multivariate-signals")
    freq_alias = pd.tseries.frequencies.to_offset(freq).freqstr
    selected_config, nf, alias, tuning_records = _fit_select_multivariate(
        config, scaled_channels, raw_by_id, channel_info, train_raws, calibrations,
        train_n, val_n, freq_alias, anchor_signal,
    )

    trainval_end = train_n + val_n
    window_metrics = []
    forecast_rows = []
    starts = range(trainval_end, n - config.horizon + 1, config.horizon)
    for window_index, start in enumerate(starts):
        history = pd.concat(
            [channel.iloc[:start][["unique_id", "ds", "y"]] for channel in scaled_channels],
            ignore_index=True,
        )
        pred = nf.predict(df=history)
        pred_col = _identify_prediction_column(pred, alias)
        for channel_index, info in enumerate(channel_info):
            nf_id = info["neuralforecast_unique_id"]
            calibration = calibrations[channel_index]
            raw = raw_by_id[nf_id]
            pred_channel = pred.loc[pred["unique_id"].astype(str).eq(nf_id)].sort_values("ds")
            y_hat = pred_channel[pred_col].to_numpy(float)[: config.horizon] * calibration["value_scale"]
            target = raw.iloc[start : start + config.horizon]
            y = target["y"].to_numpy(float)
            if len(y_hat) != len(y):
                raise RuntimeError(f"Forecast horizon length does not match target window for signal {info['signal']!r}")
            metrics = evaluate_forecast(y, y_hat, _event_config(config, calibration))
            metrics.update({"window": window_index, "start_ds": target["ds"].iloc[0], "signal": info["signal"], "source_unique_id": info["source_unique_id"]})
            window_metrics.append(metrics)
            forecast_rows.append(pd.DataFrame({
                "window": window_index, "signal": info["signal"], "source_unique_id": info["source_unique_id"],
                "ds": target["ds"].to_numpy(), "y": y, "y_hat": y_hat,
            }))

    if not window_metrics:
        raise ValueError("No complete test windows were available")

    run_name = (
        f"{config.series_id.replace('/', '_').replace(':', '_')}__{config.model}__"
        f"{config.loss}__h{config.horizon}__seed{config.seed}__mv"
    )
    output_dir = output_dir or results_root() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts = pd.concat(forecast_rows, ignore_index=True)
    metrics_frame = pd.DataFrame(window_metrics)
    forecasts.to_csv(output_dir / "forecast.csv", index=False)
    metrics_frame.to_csv(output_dir / "window_metrics.csv", index=False)
    aggregate_window_metrics(window_metrics).to_csv(output_dir / "metrics.csv", index=False)
    if config.loss != "mse":
        pd.DataFrame(tuning_records).to_csv(output_dir / "loss_grid_search.csv", index=False)

    by_signal = []
    for signal, group in metrics_frame.groupby("signal", sort=False):
        aggregated = aggregate_window_metrics(group.to_dict(orient="records"))
        aggregated.insert(0, "signal", signal)
        by_signal.append(aggregated)
    pd.concat(by_signal, ignore_index=True).to_csv(output_dir / "metrics_by_signal.csv", index=False)
    focus_rows = metrics_frame.loc[metrics_frame["signal"].eq(anchor_signal)].to_dict(orient="records")
    aggregate_window_metrics(focus_rows).to_csv(output_dir / "metrics_focus_signal.csv", index=False)

    metadata = asdict(config)
    metadata["data"] = str(config.data)
    metadata["multivariate_signals"] = list(config.multivariate_signals)
    metadata["selected_loss_hyperparameters"] = {
        "ep_quantile": selected_config.ep_quantile if config.loss == "ep" else None,
        "mep_low_quantile": selected_config.mep_low_quantile if config.loss == "mep" else None,
        "mep_high_quantile": selected_config.mep_high_quantile if config.loss == "mep" else None,
        "under_weight": selected_config.ep_under_weight if config.loss != "mse" else None,
        "over_weight": selected_config.ep_over_weight if config.loss != "mse" else None,
        "selection_metric": config.tuning_metric if config.loss != "mse" else None,
    }
    metadata.update({
        "multivariate": True, "focus_signal": anchor_signal, "frequency": str(freq),
        "neuralforecast_frequency": freq_alias, "n_series": len(channels), "n_total_per_series": n,
        "n_train": train_n, "n_val": val_n, "n_test": test_n,
        "n_test_windows": len(list(range(trainval_end, n - config.horizon + 1, config.horizon))),
        "channels": channel_info,
    })
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return output_dir


def run_experiment(config: ExperimentConfig, output_dir: Path | None = None) -> Path:
    if config.horizon < 5:
        raise ValueError("paper experiments require horizon >= 5 steps")
    if config.multivariate_signals:
        return _run_multivariate(config, output_dir)
    return _run_univariate(config, output_dir)
