#!/usr/bin/env python3
"""
Generate paper figures for the excursion-forecastability study.

Expected repository layout:
    PeakTimingAwareForecasting_forecastability/
    ├── results/
    ├── scripts/
    └── ../data/processed/
        ├── telecomts.csv
        ├── mit_supercloud_gpu_utilization_gpu_pct.csv
        └── esnet_r-nova-fd-01_delta.csv

The script recomputes Pearson predictive-structure values directly from the
processed datasets using the first 70% of each selected series, then reads the
forecasting metrics and forecasts from results/.

Usage:
    python scripts/plot_paper_results.py

Optional:
    python scripts/plot_paper_results.py \
        --results results \
        --data-root ../data/processed \
        --output results/paper_figures

Outputs both PDF (paper) and PNG (quick inspection).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = [96, 192, 336, 720]
LOSSES = ["mse", "ep", "mep"]

DATASETS = {
    "TelecomTS": {
        "file": "telecomts.csv",
        "target_id": "telecomts::normal/stationary/Zone_B/no_congestion/File::TX_Bytes",
        "scenario": "normal/stationary/Zone_B/no_congestion/File",
        "target_signal": "TX_Bytes",
        "signals": [
            "TX_Bytes",
            "RX_Bytes",
            "Estimated_UL_Buffer",
            "PRB_Utilization_DL",
            "PRB_Utilization_UL",
        ],
        "result_stem": "telecomts__normal_stationary_Zone_B_no_congestion_File__TX_Bytes",
    },
    "MIT Supercloud": {
        "file": "mit_supercloud_gpu_utilization_gpu_pct.csv",
        "target_id": "mit_gpu::67691045781943-r7343737-n685852::utilization_gpu_pct",
        "scenario": "67691045781943-r7343737-n685852",
        "target_signal": "utilization_gpu_pct",
        "signals": ["utilization_gpu_pct"],
        "result_stem": "mit_gpu__67691045781943-r7343737-n685852__utilization_gpu_pct",
    },
    "ESNet": {
        "file": "esnet_r-nova-fd-01_delta.csv",
        "alternate_files": [
            "esnet_r-nova-fd-01_delta_clean.csv",
        ],
        "target_id": "esnet::r-nova-fd-01::ifHCOutOctets",
        "scenario": "r-nova-fd-01",
        "target_signal": "ifHCOutOctets",
        "signals": [
            "ifHCOutOctets",
            "ifHCOutUcastPkts",
            "ifHCOutMulticastPkts",
            "ifHCOutBroadcastPkts",
            "ifHCInOctets",
            "ifHCInUcastPkts",
            "ifHCInMulticastPkts",
            "ifHCInBroadcastPkts",
        ],
        "result_stem": "esnet__r-nova-fd-01__ifHCOutOctets",
    },
}

UNIVARIATE_MODELS = ["dlinear", "patchtst", "itransformer", "timemixer"]
MULTIVARIATE_MODELS = ["tsmixer", "itransformer", "timemixer"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--data-root", type=Path, default=Path("../data/processed"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_figures"),
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Training prefix used for Pearson calculations.",
    )
    return p.parse_args()


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def locate_dataset(data_root: Path, spec: dict) -> Path:
    candidates = [spec["file"], *spec.get("alternate_files", [])]
    for name in candidates:
        path = data_root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} under {data_root}"
    )


def load_selected_wide(path: Path, spec: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])

    # Prefer scenario+signal matching because the processed files store
    # individual signals as separate rows.
    selected = df[
        (df["scenario"].astype(str) == spec["scenario"])
        & (df["signal"].isin(spec["signals"]))
    ].copy()

    if selected.empty:
        # MIT is simplest and can always be recovered from unique_id.
        selected = df[df["unique_id"] == spec["target_id"]].copy()

    wide = (
        selected.pivot_table(index="ds", columns="signal", values="y", aggfunc="first")
        .sort_index()
    )

    if spec["target_signal"] not in wide.columns:
        raise ValueError(
            f"{spec['target_signal']} not found in selected data from {path}"
        )

    # Require the target to exist. Auxiliary signals may have occasional gaps.
    wide = wide[wide[spec["target_signal"]].notna()]
    return wide


def predictive_structure(data_root: Path, train_fraction: float) -> pd.DataFrame:
    rows = []

    for dataset, spec in DATASETS.items():
        path = locate_dataset(data_root, spec)
        wide = load_selected_wide(path, spec)

        n_train = int(len(wide) * train_fraction)
        train = wide.iloc[:n_train].copy()
        target = spec["target_signal"]

        for h in HORIZONS:
            if len(train) <= h:
                raise ValueError(
                    f"{dataset}: training series shorter than horizon {h}"
                )

            # corr(x_t, y_{t+h}); for x == y this is target autocorrelation.
            future_y = train[target].shift(-h)

            target_r = train[target].corr(future_y)

            covariate_rs = {}
            for signal in spec["signals"]:
                if signal not in train.columns:
                    continue
                covariate_rs[signal] = train[signal].corr(future_y)

            aux = {
                k: v
                for k, v in covariate_rs.items()
                if k != target and pd.notna(v)
            }
            if aux:
                best_signal = max(aux, key=lambda k: abs(aux[k]))
                best_aux_r = aux[best_signal]
            else:
                best_signal = None
                best_aux_r = np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "horizon": h,
                    "target_r": target_r,
                    "best_aux_signal": best_signal,
                    "best_aux_r": best_aux_r,
                }
            )

    return pd.DataFrame(rows)


def read_metric(run_dir: Path, metric: str) -> float:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    row = df[df["metric"] == metric]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["mean"])


def run_dir(
    results: Path,
    dataset_stem: str,
    model: str,
    loss: str,
    horizon: int,
    multivariate: bool,
) -> Path:
    suffix = "__mv" if multivariate else ""
    return results / (
        f"{dataset_stem}__{model}__{loss}__h{horizon}__seed1{suffix}"
    )


def collect_metrics(results: Path) -> pd.DataFrame:
    rows = []

    for dataset, spec in DATASETS.items():
        for multivariate, models in [
            (False, UNIVARIATE_MODELS),
            (True, MULTIVARIATE_MODELS),
        ]:
            # MIT has no multivariate experiment.
            if dataset == "MIT Supercloud" and multivariate:
                continue

            for model in models:
                for h in HORIZONS:
                    for loss in LOSSES:
                        d = run_dir(
                            results,
                            spec["result_stem"],
                            model,
                            loss,
                            h,
                            multivariate,
                        )
                        if not d.exists():
                            continue

                        rows.append(
                            {
                                "dataset": dataset,
                                "mode": "multivariate" if multivariate else "univariate",
                                "model": model,
                                "loss": loss,
                                "horizon": h,
                                "event_macro_f1": read_metric(d, "event_macro_f1"),
                                "peak_f1": read_metric(d, "peak_f1"),
                                "trough_f1": read_metric(d, "trough_f1"),
                                "mse": read_metric(d, "mse"),
                                "mae": read_metric(d, "mae"),
                            }
                        )

    return pd.DataFrame(rows)


def plot_correlation_by_horizon(corr: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for dataset in DATASETS:
        d = corr[corr["dataset"] == dataset].sort_values("horizon")
        ax.plot(
            d["horizon"],
            d["target_r"],
            marker="o",
            linewidth=1.8,
            label=dataset,
        )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel(r"Pearson $r(y_t, y_{t+h})$")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "correlation_by_horizon")


def plot_target_and_best_covariate(corr: pd.DataFrame, output: Path):
    # Useful for TelecomTS and ESNet; MIT has no companion telemetry.
    for dataset in ["TelecomTS", "ESNet"]:
        d = corr[corr["dataset"] == dataset].sort_values("horizon")

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(
            d["horizon"],
            d["target_r"],
            marker="o",
            linewidth=1.8,
            label="Target history",
        )
        ax.plot(
            d["horizon"],
            d["best_aux_r"].abs(),
            marker="s",
            linewidth=1.8,
            label="Strongest covariate (absolute r)",
        )

        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel("Pearson correlation")
        ax.set_xticks(HORIZONS)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(
            fig,
            output,
            f"{dataset.lower().replace(' ', '_')}_predictive_structure",
        )


def plot_event_f1_by_horizon(metrics: pd.DataFrame, output: Path):
    """
    One figure per dataset/mode. Each line is a loss and each point is the
    mean event macro-F1 across the models available in that mode.

    The full per-model results should still be reported in a table; this plot
    is for the broad trend.
    """
    for (dataset, mode), d in metrics.groupby(["dataset", "mode"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))

        for loss in LOSSES:
            x = (
                d[d["loss"] == loss]
                .groupby("horizon", as_index=False)["event_macro_f1"]
                .mean()
                .sort_values("horizon")
            )
            if x.empty:
                continue
            ax.plot(
                x["horizon"],
                x["event_macro_f1"],
                marker="o",
                linewidth=1.8,
                label=loss.upper(),
            )

        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel("Mean event macro-F1")
        ax.set_xticks(HORIZONS)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        fig.tight_layout()

        stem = (
            f"{dataset.lower().replace(' ', '_')}_"
            f"{mode}_event_f1_by_horizon"
        )
        save_figure(fig, output, stem)


def plot_correlation_vs_event_f1(
    corr: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
):
    """
    Directly ties predictive structure to excursion forecasting performance.
    Each point is one model/loss/horizon experiment. The x value is the target
    Pearson correlation for that dataset/horizon.
    """
    merged = metrics.merge(
        corr[["dataset", "horizon", "target_r"]],
        on=["dataset", "horizon"],
        how="inner",
    )
    merged = merged[
        merged["event_macro_f1"].notna() & merged["target_r"].notna()
    ].copy()

    # Separate figure per loss keeps the figure readable and avoids turning
    # architecture/loss/dataset into an unreadable marker legend.
    for loss in LOSSES:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        dloss = merged[merged["loss"] == loss]

        for dataset in DATASETS:
            d = dloss[dloss["dataset"] == dataset]
            if d.empty:
                continue
            ax.scatter(
                d["target_r"],
                d["event_macro_f1"],
                s=38,
                alpha=0.75,
                label=dataset,
            )

        # Add a simple least-squares trend only if enough finite points exist.
        finite = dloss[
            np.isfinite(dloss["target_r"])
            & np.isfinite(dloss["event_macro_f1"])
        ]
        if len(finite) >= 3 and finite["target_r"].nunique() > 1:
            x = finite["target_r"].to_numpy()
            y = finite["event_macro_f1"].to_numpy()
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m * xs + b, linewidth=1.2)

            r = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.03,
                0.95,
                f"Across runs: r = {r:.2f}",
                transform=ax.transAxes,
                va="top",
            )

        ax.set_xlabel(r"Target Pearson $r(y_t, y_{t+h})$")
        ax.set_ylabel("Event macro-F1")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output, f"correlation_vs_event_f1_{loss}")


def read_forecast_set(
    results: Path,
    dataset_stem: str,
    model: str,
    horizon: int,
    window: int,
    multivariate: bool = False,
):
    frames = {}
    for loss in LOSSES:
        d = run_dir(
            results,
            dataset_stem,
            model,
            loss,
            horizon,
            multivariate,
        )
        path = d / "forecast.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        f = pd.read_csv(path)
        f["ds"] = pd.to_datetime(f["ds"])
        f = f[f["window"] == window].copy()
        if f.empty:
            raise ValueError(
                f"No window {window} in {path}"
            )
        frames[loss] = f

    return frames


def plot_forecast_example(
    results: Path,
    output: Path,
    dataset: str,
    model: str,
    horizon: int,
    window: int,
    ylabel: str,
    stem: str,
):
    spec = DATASETS[dataset]
    frames = read_forecast_set(
        results,
        spec["result_stem"],
        model,
        horizon,
        window,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    actual = frames["mse"]
    ax.plot(
        actual["ds"],
        actual["y"],
        linewidth=2.0,
        label="Actual",
    )

    for loss in LOSSES:
        f = frames[loss]
        ax.plot(
            f["ds"],
            f["y_hat"],
            linewidth=1.35,
            label=loss.upper(),
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, output, stem)


def write_summary_tables(
    corr: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
):
    output.mkdir(parents=True, exist_ok=True)

    corr.to_csv(output / "predictive_structure.csv", index=False)
    metrics.to_csv(output / "forecast_metrics_selected_runs.csv", index=False)

    # Loss-level summary by dataset/horizon/mode.
    summary = (
        metrics.groupby(
            ["dataset", "mode", "horizon", "loss"],
            as_index=False,
        )
        .agg(
            mean_event_macro_f1=("event_macro_f1", "mean"),
            mean_peak_f1=("peak_f1", "mean"),
            mean_trough_f1=("trough_f1", "mean"),
            mean_mse=("mse", "mean"),
            n_models=("model", "nunique"),
        )
    )
    summary.to_csv(output / "forecast_metrics_summary.csv", index=False)


def main():
    args = parse_args()
    args.results = args.results.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()

    print(f"Results: {args.results}")
    print(f"Data:    {args.data_root}")
    print(f"Output:  {args.output}")

    corr = predictive_structure(args.data_root, args.train_fraction)
    metrics = collect_metrics(args.results)

    if metrics.empty:
        raise RuntimeError(
            "No matching MSE/EP/MEP result directories were found."
        )

    write_summary_tables(corr, metrics, args.output)

    # Main quantitative plots.
    plot_correlation_by_horizon(corr, args.output)
    plot_target_and_best_covariate(corr, args.output)
    plot_event_f1_by_horizon(metrics, args.output)
    plot_correlation_vs_event_f1(corr, metrics, args.output)

    # Representative qualitative cases selected from the current results.
    #
    # 1. TelecomTS: MEP recovers some excursions at h=336 but also produces
    #    considerably more excursion activity than MSE/EP.
    plot_forecast_example(
        args.results,
        args.output,
        dataset="TelecomTS",
        model="dlinear",
        horizon=336,
        window=1,
        ylabel="TX_Bytes",
        stem="telecomts_dlinear_h336_window1",
    )

    # 2. MIT: all objectives stay near the dominant high-utilization regime
    #    while the actual series contains an abrupt utilization trough.
    plot_forecast_example(
        args.results,
        args.output,
        dataset="MIT Supercloud",
        model="timemixer",
        horizon=720,
        window=0,
        ylabel="GPU utilization (%)",
        stem="mit_timemixer_h720_window0",
    )

    # 3. ESNet: useful qualitative MEP case. Window 36 contains the trough
    #    whose timing MEP places correctly while MSE/EP are later.
    plot_forecast_example(
        args.results,
        args.output,
        dataset="ESNet",
        model="timemixer",
        horizon=96,
        window=36,
        ylabel="ifHCOutOctets delta",
        stem="esnet_timemixer_h96_window36",
    )

    print("\nGenerated:")
    for p in sorted(args.output.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
