import numpy as np
import pandas as pd

from ptaf.signal import centered_moving_average, longest_contiguous_segment, robust_residual_scale


def test_centered_moving_average_preserves_length():
    x = np.arange(20, dtype=float)
    assert centered_moving_average(x, 5).shape == x.shape


def test_robust_residual_scale_positive_for_spiky_signal():
    x = np.zeros(101)
    x[50] = 10
    assert robust_residual_scale(x, 11) > 0


def test_longest_contiguous_segment_does_not_interpolate_gap():
    ds = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:05"])
    df = pd.DataFrame({"ds": ds, "y": [1, 2, 3]})
    out = longest_contiguous_segment(df, pd.Timedelta(seconds=1))
    assert len(out) == 2
