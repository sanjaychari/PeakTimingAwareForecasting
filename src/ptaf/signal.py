from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def odd_window(window: int, n: int) -> int:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else max(1, n - 1)
    return max(1, window)


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(values) == 0:
        return values.copy()
    w = odd_window(window, len(values))
    if w == 1:
        return values.copy()
    radius = w // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(padded, kernel, mode="valid")


def robust_residual_scale(values: np.ndarray, trend_window: int) -> float:
    values = np.asarray(values, dtype=float)
    trend = centered_moving_average(values, trend_window)
    residual = values - trend
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(residual))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(values))
    return max(scale, EPS)


def regularize_series(
    df: pd.DataFrame,
    *,
    interval: str,
    value_col: str = "y",
) -> pd.DataFrame:
    """Resample by mean without fabricating missing samples.

    Empty bins remain missing and are removed. The caller can then select a
    contiguous segment explicitly rather than interpolating across gaps.
    """
    if df.empty:
        return df.copy()
    out = (
        df.set_index("ds")[[value_col]]
        .resample(interval)
        .mean()
        .dropna()
        .reset_index()
    )
    return out


def longest_contiguous_segment(df: pd.DataFrame, freq: pd.Timedelta) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)
    if len(ordered) <= 1:
        return ordered
    delta = ordered["ds"].diff()
    groups = delta.ne(freq).cumsum()
    sizes = groups.value_counts()
    keep = sizes.idxmax()
    return ordered.loc[groups.eq(keep)].reset_index(drop=True)
