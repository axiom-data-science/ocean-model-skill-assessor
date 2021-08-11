import numpy as np
import pandas as pd

from skill_assessor.plot import time_series


def test_time_series():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.view(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'FAKE_SAMPLES': np.sin(sample_times.view(int))
    }, index=sample_times)

    time_series.plot(reference, sample, 'test')