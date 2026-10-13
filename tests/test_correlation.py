import numpy as np
import pandas as pd

from ptaf.correlation import analyze_correlations


def _frame():
    n = 2000
    ds = pd.date_range("2026-01-01", periods=n, freq="s")
    target = np.arange(n, dtype=float)
    aux = target * 2.0
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": "d::s::target",
                    "ds": ds,
                    "y": target,
                    "dataset": "d",
                    "signal": "target",
                    "scenario": "s",
                }
            ),
            pd.DataFrame(
                {
                    "unique_id": "d::s::aux",
                    "ds": ds,
                    "y": aux,
                    "dataset": "d",
                    "signal": "aux",
                    "scenario": "s",
                }
            ),
        ],
        ignore_index=True,
    )


def test_lead_correlations_use_training_prefix_and_aligned_signals():
    result = analyze_correlations(
        _frame(),
        "d::s::target",
        (96, 192),
        signals=("target", "aux"),
        train_fraction=0.7,
    )
    assert set(result["horizon"]) == {96, 192}
    assert set(result["signal"]) == {"target", "aux"}
    assert np.allclose(result["pearson_r"], 1.0)
    assert set(result.loc[result["signal"].eq("target"), "correlation_type"]) == {"autocorrelation"}
    assert set(result.loc[result["signal"].eq("aux"), "correlation_type"]) == {"cross_correlation"}
