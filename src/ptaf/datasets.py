from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import data_root
from .signal import longest_contiguous_segment

TELECOM_KPIS = (
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
)



ESNET_SIGNALS = (
    "ifHCOutOctets",
    "ifHCOutUcastPkts",
    "ifHCOutMulticastPkts",
    "ifHCOutBroadcastPkts",
    "ifHCInOctets",
    "ifHCInUcastPkts",
    "ifHCInMulticastPkts",
    "ifHCInBroadcastPkts",
)

MIT_GPU_METRICS = (
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_MiB",
    "power_draw_W",
)


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    source: str
    frequency: pd.Timedelta


def _write(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def prepare_telecomts(
    root: Path | None = None,
    output: Path | None = None,
    *,
    scope: str = "normal",
    kpis: tuple[str, ...] = TELECOM_KPIS,
    resample_ms: int = 1000,
) -> PreparedDataset:
    root = root or data_root() / "raw" / "telecomts"
    output = output or data_root() / "processed" / "telecomts.csv"
    if scope not in {"normal", "anomalous", "all"}:
        raise ValueError("scope must be one of: normal, anomalous, all")

    search_roots: list[Path] = []
    if scope in {"normal", "all"}:
        search_roots.append(root / "normal")
    if scope in {"anomalous", "all"}:
        search_roots.append(root / "anomalous")

    files: list[Path] = []
    for base in search_roots:
        if base.exists():
            files.extend(sorted(base.glob("**/raw/metrics.csv")))
    if not files:
        raise FileNotFoundError(f"No TelecomTS raw metrics.csv files found below {root}")

    interval = pd.Timedelta(milliseconds=resample_ms)
    rows: list[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path, usecols=lambda c: c == "timestamp" or c in kpis)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
        scenario = path.relative_to(root).as_posix().removesuffix("/raw/metrics.csv")
        for kpi in kpis:
            if kpi not in raw.columns:
                continue
            series = raw[["timestamp", kpi]].rename(columns={"timestamp": "ds", kpi: "y"})
            series["y"] = pd.to_numeric(series["y"], errors="coerce")
            series = series.dropna(subset=["y"]).set_index("ds").resample(interval).mean().dropna().reset_index()
            series = longest_contiguous_segment(series, interval)
            if len(series) < 10:
                continue
            series.insert(0, "unique_id", f"telecomts::{scenario}::{kpi}")
            series["dataset"] = "telecomts"
            series["signal"] = kpi
            series["scenario"] = scenario
            rows.append(series)

    if not rows:
        raise ValueError("TelecomTS preparation produced no usable series")
    frame = pd.concat(rows, ignore_index=True)
    _write(frame, output)
    return PreparedDataset(frame=frame, source=str(root), frequency=interval)


def prepare_mit_supercloud_gpu(
    root: Path | None = None,
    output: Path | None = None,
    *,
    metric: str = "utilization_gpu_pct",
    resample_ms: int = 1000,
) -> PreparedDataset:
    if metric not in MIT_GPU_METRICS:
        raise ValueError(f"Unsupported MIT GPU metric {metric!r}; choose from {MIT_GPU_METRICS}")
    root = root or data_root() / "raw" / "mit_supercloud" / "gpu_labeled_sample"
    output = output or data_root() / "processed" / f"mit_supercloud_gpu_{metric}.csv"
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No MIT Supercloud GPU CSV files found in {root}")

    interval = pd.Timedelta(milliseconds=resample_ms)
    rows: list[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path, usecols=["timestamp", "gpu_index", metric])
        raw["ds"] = pd.to_datetime(raw["timestamp"], unit="s", errors="coerce")
        raw[metric] = pd.to_numeric(raw[metric], errors="coerce")
        raw = raw.dropna(subset=["ds", metric])
        # A job-node file may contain multiple GPUs. Mean utilization across the
        # GPUs gives one workload-level signal while retaining simultaneous drops.
        series = (
            raw.set_index("ds")[[metric]]
            .resample(interval)
            .mean()
            .dropna()
            .rename(columns={metric: "y"})
            .reset_index()
        )
        series = longest_contiguous_segment(series, interval)
        if len(series) < 10:
            continue
        series.insert(0, "unique_id", f"mit_gpu::{path.stem}::{metric}")
        series["dataset"] = "mit_supercloud_gpu"
        series["signal"] = metric
        series["scenario"] = path.stem
        rows.append(series)

    if not rows:
        raise ValueError("MIT Supercloud GPU preparation produced no usable series")
    frame = pd.concat(rows, ignore_index=True)
    _write(frame, output)
    return PreparedDataset(frame=frame, source=str(root), frequency=interval)



def prepare_esnet(
    raw_path: Path,
    output: Path | None = None,
    *,
    signals: tuple[str, ...] = ESNET_SIGNALS,
    scenario: str = "r-nova-fd-01",
) -> PreparedDataset:
    """Convert a wide ESNet *delta-counter* CSV to aligned long format.

    The input must already contain per-interval deltas generated from the raw
    IF-MIB cumulative counters. We intentionally keep only the longest exact
    regular segment so forecasting windows never cross telemetry outages, DST
    ambiguity, or other collection gaps.
    """
    raw = pd.read_csv(raw_path)
    if "date" not in raw.columns:
        raise ValueError("ESNet delta CSV must contain a 'date' column")
    missing = [signal for signal in signals if signal not in raw.columns]
    if missing:
        raise ValueError(f"ESNet delta CSV is missing requested signals: {missing}")

    raw["ds"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    for signal in signals:
        raw[signal] = pd.to_numeric(raw[signal], errors="coerce")
    raw = raw.dropna(subset=list(signals)).reset_index(drop=True)
    if len(raw) < 2:
        raise ValueError("ESNet delta CSV has fewer than two complete rows")
    if (raw[list(signals)] < 0).any().any():
        raise ValueError("ESNet delta CSV contains negative counter deltas")

    deltas = raw["ds"].diff().dropna()
    mode = deltas.mode()
    interval = mode.iloc[0] if len(mode) else deltas.median()
    if interval <= pd.Timedelta(0):
        raise ValueError("Could not infer a positive ESNet sampling interval")

    # Select the longest common contiguous segment once, before expanding to
    # long format. This guarantees exact timestamp alignment across channels.
    marker = raw["ds"].diff().ne(interval).cumsum()
    sizes = raw.groupby(marker, sort=False).size()
    best = sizes.idxmax()
    clean = raw.loc[marker.eq(best), ["ds", *signals]].copy().reset_index(drop=True)
    if len(clean) < 10:
        raise ValueError("Longest contiguous ESNet segment is too short")

    rows: list[pd.DataFrame] = []
    for signal in signals:
        series = clean[["ds", signal]].rename(columns={signal: "y"}).copy()
        series.insert(0, "unique_id", f"esnet::{scenario}::{signal}")
        series["dataset"] = "esnet"
        series["signal"] = signal
        series["scenario"] = scenario
        rows.append(series)

    frame = pd.concat(rows, ignore_index=True)
    output = output or data_root() / "processed" / f"esnet_{scenario}_delta.csv"
    _write(frame, output)
    return PreparedDataset(frame=frame, source=str(raw_path), frequency=interval)

def load_prepared(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"unique_id", "ds", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {sorted(missing)}")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    frame["y"] = pd.to_numeric(frame["y"], errors="raise")
    return frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def infer_frequency(series: pd.DataFrame) -> pd.Timedelta:
    delta = series.sort_values("ds")["ds"].diff().dropna()
    if delta.empty:
        raise ValueError("Need at least two samples to infer frequency")
    mode = delta.mode()
    return mode.iloc[0] if len(mode) else delta.median()


def summarize_series(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    prominence_fraction: float = 0.10,
    prominence_window: int = 31,
    event_distance: int = 3,
) -> pd.DataFrame:
    from scipy.signal import find_peaks

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    records = []
    wlen = max(3, int(prominence_window))
    if wlen % 2 == 0:
        wlen += 1

    for uid, group in frame.groupby("unique_id", sort=True):
        y = group["y"].to_numpy(float)
        train_n = max(1, int(np.floor(len(y) * train_fraction)))
        train_y = y[:train_n]
        q01_train = float(np.quantile(train_y, 0.01))
        q99_train = float(np.quantile(train_y, 0.99))
        prominence_threshold = prominence_fraction * (q99_train - q01_train)
        peaks, peak_props = find_peaks(
            y,
            prominence=prominence_threshold,
            wlen=wlen,
            distance=max(1, int(event_distance)),
        )
        troughs, trough_props = find_peaks(
            -y,
            prominence=prominence_threshold,
            wlen=wlen,
            distance=max(1, int(event_distance)),
        )
        n_events = len(peaks) + len(troughs)
        records.append(
            {
                "unique_id": uid,
                "n": len(y),
                "median": float(np.median(y)),
                "q01": float(np.quantile(y, 0.01)),
                "q99": float(np.quantile(y, 0.99)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "peak_prominence_threshold": prominence_threshold,
                "trough_prominence_threshold": prominence_threshold,
                "prominence_fraction": prominence_fraction,
                "peak_count": int(len(peaks)),
                "trough_count": int(len(troughs)),
                "events_per_1000": float(1000.0 * n_events / max(len(y), 1)),
                "max_peak_prominence": float(np.max(peak_props["prominences"])) if len(peaks) else 0.0,
                "max_trough_prominence": float(np.max(trough_props["prominences"])) if len(troughs) else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)
