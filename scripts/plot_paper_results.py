#!/usr/bin/env python3
"""
Generate final two-column paper figures for the systems excursion
forecastability experiments.

The script produces three large, vertically stacked multi-panel figures:

  1. predictive_structure.pdf
     Three full-width rows: TelecomTS, MIT Supercloud, ESNet.

  2. event_f1_by_horizon.pdf
     Three full-width rows: event macro-F1 versus forecast horizon.
     MSE, EP, and MEP are shown separately. Each point is the mean across
     the available model/configuration runs for that dataset/horizon/loss.
     Detailed model- and mode-specific results remain in the CSV tables.

  3. representative_forecasts.pdf
     Three full-width rows: representative TelecomTS, MIT, and ESNet
     forecast windows.

The script also writes CSVs for LaTeX tables. No PNGs are produced.

Usage:
    python scripts/plot_paper_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
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
        "alternate_files": ["esnet_r-nova-fd-01_delta_clean.csv"],
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

FINAL_PDFS = {
    "predictive_structure.pdf",
    "event_f1_by_horizon.pdf",
    "representative_forecasts.pdf",
}

FINAL_CSVS = {
    "predictive_structure.csv",
    "forecast_metrics_by_run.csv",
    "forecast_metrics_summary.csv",
}


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
        help="Training fraction used for Pearson calculations.",
    )
    return p.parse_args()


def configure_matplotlib():
    # These figures are intended for figure* in a two-column paper.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_output_dir(output: Path):
    output.mkdir(parents=True, exist_ok=True)

    # Remove stale paper-figure assets produced by older versions of this
    # script, but do not touch the experiment result directories.
    for path in output.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".csv"}:
            path.unlink()


def save_pdf(fig, output: Path, filename: str):
    fig.savefig(output / filename, bbox_inches="tight")
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

    selected = pd.DataFrame()

    if {"scenario", "signal"}.issubset(df.columns):
        selected = df[
            (df["scenario"].astype(str) == spec["scenario"])
            & (df["signal"].isin(spec["signals"]))
        ].copy()

    if selected.empty and "unique_id" in df.columns:
        selected = df[
            df["unique_id"].astype(str) == spec["target_id"]
        ].copy()

    if selected.empty:
        raise ValueError(
            f"Could not locate selected series {spec['target_id']} in {path}"
        )

    wide = (
        selected.pivot_table(
            index="ds",
            columns="signal",
            values="y",
            aggfunc="first",
        )
        .sort_index()
    )

    target = spec["target_signal"]
    if target not in wide.columns:
        raise ValueError(f"{target} not found in selected rows from {path}")

    return wide[wide[target].notna()]


def predictive_structure(data_root: Path, train_fraction: float) -> pd.DataFrame:
    rows = []

    for dataset, spec in DATASETS.items():
        path = locate_dataset(data_root, spec)
        wide = load_selected_wide(path, spec)

        n_train = int(len(wide) * train_fraction)
        train = wide.iloc[:n_train].copy()
        target = spec["target_signal"]

        for horizon in HORIZONS:
            future_y = train[target].shift(-horizon)
            target_r = train[target].corr(future_y)

            aux_rs = {}
            for signal in spec["signals"]:
                if signal == target or signal not in train.columns:
                    continue

                r = train[signal].corr(future_y)
                if pd.notna(r):
                    aux_rs[signal] = r

            if aux_rs:
                best_aux_signal = max(aux_rs, key=lambda s: abs(aux_rs[s]))
                best_aux_r = aux_rs[best_aux_signal]
            else:
                best_aux_signal = None
                best_aux_r = np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "target_r": target_r,
                    "best_aux_signal": best_aux_signal,
                    "best_aux_r": best_aux_r,
                }
            )

    return pd.DataFrame(rows)


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


def read_metric_record(
    run: Path,
    metric: str,
    multivariate: bool = False,
) -> dict:
    path = run / (
        "metrics_focus_signal.csv" if multivariate else "metrics.csv"
    )

    empty = {
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "n_valid": np.nan,
    }

    if not path.exists():
        return empty

    df = pd.read_csv(path)
    row = df[df["metric"] == metric]

    if row.empty:
        return empty

    row = row.iloc[0]

    return {
        "mean": float(row["mean"]) if pd.notna(row.get("mean")) else np.nan,
        "median": float(row["median"]) if pd.notna(row.get("median")) else np.nan,
        "std": float(row["std"]) if pd.notna(row.get("std")) else np.nan,
        "n_valid": (
            int(row["n_valid"])
            if "n_valid" in row and pd.notna(row["n_valid"])
            else np.nan
        ),
    }


def collect_metrics(results: Path) -> pd.DataFrame:
    metric_names = [
        "event_macro_f1",
        "peak_f1",
        "trough_f1",
        "mse",
        "mae",
        "event_timing_mae_steps",
        "event_magnitude_mae",
    ]

    rows = []

    for dataset, spec in DATASETS.items():
        settings = [
            (False, UNIVARIATE_MODELS),
            (True, MULTIVARIATE_MODELS),
        ]

        for multivariate, models in settings:
            if dataset == "MIT Supercloud" and multivariate:
                continue

            for model in models:
                for horizon in HORIZONS:
                    for loss in LOSSES:
                        run = run_dir(
                            results,
                            spec["result_stem"],
                            model,
                            loss,
                            horizon,
                            multivariate,
                        )

                        if not run.exists():
                            continue

                        row = {
                            "dataset": dataset,
                            "mode": (
                                "multivariate"
                                if multivariate
                                else "univariate"
                            ),
                            "model": model,
                            "loss": loss,
                            "horizon": horizon,
                        }

                        for metric in metric_names:
                            stats = read_metric_record(
                                run,
                                metric,
                                multivariate=multivariate,
                            )
                            for stat, value in stats.items():
                                row[f"{metric}_{stat}"] = value

                        rows.append(row)

    return pd.DataFrame(rows)


def panel_title(ax, label: str, title: str):
    ax.text(
        0.0,
        1.03,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )


def plot_predictive_structure(corr: pd.DataFrame, output: Path):
    """
    One dataset per full-width row.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.3, 8.2),
        sharex=True,
    )

    panels = [
        ("a", "TelecomTS"),
        ("b", "MIT Supercloud"),
        ("c", "ESNet"),
    ]

    for ax, (label, dataset) in zip(axes, panels):
        d = corr[corr["dataset"] == dataset].sort_values("horizon")

        ax.plot(
            d["horizon"],
            d["target_r"],
            marker="o",
            label="Target history",
        )

        if dataset != "MIT Supercloud":
            ax.plot(
                d["horizon"],
                d["best_aux_r"].abs(),
                marker="s",
                label="Strongest covariate",
            )

        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylabel("Pearson correlation")
        ax.set_xticks(HORIZONS)
        panel_title(ax, label, dataset)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Forecast horizon (steps)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.08,
        top=0.93,
        hspace=0.42,
    )

    save_pdf(fig, output, "predictive_structure.pdf")


def plot_event_f1_by_horizon(metrics: pd.DataFrame, output: Path):
    """
    Broad loss-level view.

    Each point is the mean of the run-level mean event macro-F1 values for
    all available model/configuration runs at that dataset/horizon/loss.

    The detailed model, mode, and test-window results are kept in CSVs and
    should be presented in LaTeX tables.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.3, 8.2),
        sharex=True,
    )

    panels = [
        ("a", "TelecomTS"),
        ("b", "MIT Supercloud"),
        ("c", "ESNet"),
    ]

    for ax, (label, dataset) in zip(axes, panels):
        dset = metrics[metrics["dataset"] == dataset]

        for loss in LOSSES:
            dloss = dset[dset["loss"] == loss]

            summary = (
                dloss.groupby("horizon", as_index=False)
                .agg(
                    mean_f1=("event_macro_f1_mean", "mean"),
                    n_runs=("event_macro_f1_mean", "count"),
                )
                .sort_values("horizon")
            )

            if summary.empty:
                continue

            ax.plot(
                summary["horizon"],
                summary["mean_f1"],
                marker="o",
                label=loss.upper(),
            )

        ax.set_ylabel("Mean event macro-F1")
        ax.set_ylim(bottom=0)
        ax.set_xticks(HORIZONS)
        panel_title(ax, label, dataset)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Forecast horizon (steps)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.08,
        top=0.93,
        hspace=0.42,
    )

    save_pdf(fig, output, "event_f1_by_horizon.pdf")


def read_forecast_set(
    results: Path,
    dataset_stem: str,
    model: str,
    horizon: int,
    window: int,
    multivariate: bool = False,
) -> dict[str, pd.DataFrame]:
    frames = {}

    for loss in LOSSES:
        run = run_dir(
            results,
            dataset_stem,
            model,
            loss,
            horizon,
            multivariate,
        )

        path = run / "forecast.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame["ds"] = pd.to_datetime(frame["ds"])
        frame = frame[frame["window"] == window].copy()

        if frame.empty:
            raise ValueError(f"No window {window} in {path}")

        frames[loss] = frame

    return frames


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.DateFormatter("%H:%M")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def plot_representative_forecasts(results: Path, output: Path):
    """
    One representative forecast per full-width row.
    """
    examples = [
        {
            "panel": "a",
            "dataset": "TelecomTS",
            "model": "dlinear",
            "horizon": 336,
            "window": 1,
            "ylabel": "TX_Bytes",
            "title": "TelecomTS — DLinear, h=336",
        },
        {
            "panel": "b",
            "dataset": "MIT Supercloud",
            "model": "timemixer",
            "horizon": 720,
            "window": 0,
            "ylabel": "GPU utilization (%)",
            "title": "MIT Supercloud — TimeMixer, h=720",
        },
        {
            "panel": "c",
            "dataset": "ESNet",
            "model": "timemixer",
            "horizon": 96,
            "window": 36,
            "ylabel": "ifHCOutOctets delta",
            "title": "ESNet — TimeMixer, h=96",
        },
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.3, 8.8),
    )

    legend_handles = None
    legend_labels = None

    for ax, example in zip(axes, examples):
        spec = DATASETS[example["dataset"]]

        frames = read_forecast_set(
            results,
            spec["result_stem"],
            example["model"],
            example["horizon"],
            example["window"],
        )

        actual = frames["mse"]

        ax.plot(
            actual["ds"],
            actual["y"],
            linewidth=2.4,
            label="Actual",
        )

        for loss in LOSSES:
            frame = frames[loss]

            ax.plot(
                frame["ds"],
                frame["y_hat"],
                linewidth=1.7,
                label=loss.upper(),
            )

        panel_title(
            ax,
            example["panel"],
            example["title"],
        )
        format_time_axis(ax)
        ax.grid(axis="y", alpha=0.20)

        if example["dataset"] == "TelecomTS":
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{value / 1e5:g}")
            )
            ax.set_ylabel(r"TX_Bytes ($\times 10^5$)")
        elif example["dataset"] == "ESNet":
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda value, _: f"{value / 1e11:g}")
            )
            ax.set_ylabel(r"ifHCOutOctets delta ($\times 10^{11}$)")
        else:
            ax.set_ylabel(example["ylabel"])

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("Time")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.08,
        top=0.93,
        hspace=0.48,
    )

    save_pdf(fig, output, "representative_forecasts.pdf")


def write_table_data(
    corr: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
):
    corr.to_csv(
        output / "predictive_structure.csv",
        index=False,
    )

    metrics.to_csv(
        output / "forecast_metrics_by_run.csv",
        index=False,
    )

    summary = (
        metrics.groupby(
            ["dataset", "mode", "horizon", "loss"],
            as_index=False,
        )
        .agg(
            model_count=("model", "nunique"),
            event_macro_f1_model_mean=("event_macro_f1_mean", "mean"),
            event_macro_f1_model_std=("event_macro_f1_mean", "std"),
            peak_f1_model_mean=("peak_f1_mean", "mean"),
            trough_f1_model_mean=("trough_f1_mean", "mean"),
            mse_model_mean=("mse_mean", "mean"),
            mae_model_mean=("mae_mean", "mean"),
        )
    )

    summary.to_csv(
        output / "forecast_metrics_summary.csv",
        index=False,
    )


def main():
    args = parse_args()

    results = args.results.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()

    configure_matplotlib()
    prepare_output_dir(output)

    print(f"Results: {results}")
    print(f"Data:    {data_root}")
    print(f"Output:  {output}")

    corr = predictive_structure(
        data_root,
        args.train_fraction,
    )

    metrics = collect_metrics(results)

    if metrics.empty:
        raise RuntimeError(
            "No matching MSE/EP/MEP paper runs were found under results/."
        )

    write_table_data(
        corr,
        metrics,
        output,
    )

    plot_predictive_structure(
        corr,
        output,
    )

    plot_event_f1_by_horizon(
        metrics,
        output,
    )

    plot_representative_forecasts(
        results,
        output,
    )

    print("\nGenerated paper assets:")
    for filename in sorted(FINAL_PDFS | FINAL_CSVS):
        path = output / filename
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
