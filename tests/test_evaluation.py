import numpy as np
from ptaf.evaluation import EventConfig, evaluate_forecast


def cfg():
    return EventConfig(peak_prominence=3.0, trough_prominence=3.0, prominence_window=31, distance=3, tolerance=3)


def test_event_evaluation_is_bidirectional():
    y=np.zeros(31); y[10]=10.0; y[20]=-10.0
    result=evaluate_forecast(y,y.copy(),cfg())
    assert result['peak_recall']==1.0
    assert result['trough_recall']==1.0


def test_quiet_window_event_f1_is_undefined_not_perfect():
    y=np.zeros(31)
    result=evaluate_forecast(y,y.copy(),cfg())
    assert np.isnan(result['event_macro_f1'])
