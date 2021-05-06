import numpy as np
import pandas as pd

from skill_assessor import stats


def test_align():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'FAKE_SAMPLES': np.sin(sample_times.astype(int))
    }, index=sample_times)

    aligned_signals = stats._align(reference, sample)

    assert isinstance(aligned_signals, pd.DataFrame)
    assert aligned_signals.shape == (17, 2)
    assert np.isclose(aligned_signals['sample'].mean(), 0.026802742458784518)
    assert np.isclose(aligned_signals['reference'].mean(), 0.023065018092923745)


def test_bias():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    bias = stats.compute_bias(reference, sample)

    assert np.isclose(bias, 0.0037377243658607757)


def test_correlation_coefficient():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    corr_coef = stats.compute_correlation_coefficient(reference, sample)

    assert np.isclose(corr_coef, 0.9910875774653888)


def test_index_of_agreement():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    ioa = stats.compute_index_of_agreement(reference, sample)

    assert np.isclose(ioa, 0.9675229729784884)


def test_mean_square_error():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    mse = stats.compute_mean_square_error(reference, sample, centered=False)

    assert np.isclose(mse, 0.019124891068518297)


def test_mean_square_error_centered():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    mse = stats.compute_mean_square_error(reference, sample, centered=True)

    assert np.isclose(mse, 0.019110920485083147)


def test_murphy_skill_score():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    mss = stats.compute_murphy_skill_score(reference, sample)

    assert np.isclose(mss, 0.9617732386418576)


def test_root_mean_square_error():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    rmse = stats.compute_root_mean_square_error(reference, sample)

    assert np.isclose(rmse, 0.13824225289354608)


def test_standard_deviation():
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    reference = pd.DataFrame({
        'reference': np.sin(ref_times.astype(int))
    }, index=ref_times)

    sample_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    sample = pd.DataFrame({
        'sample': np.sin(sample_times.astype(int))
    }, index=sample_times)

    std = stats.compute_standard_deviation(sample)

    assert np.isclose(std, 0.690945)
