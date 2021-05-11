import numpy as np
import pandas as pd

from skill_assessor import stats


class TestStats:
    ref_times = pd.date_range(start='2000-12-30', end='2001-01-03', freq='6H')
    obs = pd.DataFrame({
        'obs': np.sin(ref_times.astype(int))
    }, index=ref_times)

    model_times = pd.date_range(start='2000-12-28', end='2001-01-04', freq='D')
    model = pd.DataFrame({
        'model': np.sin(model_times.astype(int))
    }, index=model_times)

    def test_align(self):
        model = pd.DataFrame({
            'FAKE_SAMPLES': np.sin(self.model_times.astype(int))
        }, index=self.model_times)

        aligned_signals = stats._align(self.obs, model)

        assert isinstance(aligned_signals, pd.DataFrame)
        assert aligned_signals.shape == (17, 2)
        assert np.isclose(aligned_signals['model'].mean(), 0.026802742458784518)
        assert np.isclose(aligned_signals['obs'].mean(), 0.023065018092923745)

    def test_bias(self):
        bias = stats.compute_bias(self.obs, self.model)

        assert np.isclose(bias, 0.0037377243658607757)

    def test_correlation_coefficient(self):
        corr_coef = stats.compute_correlation_coefficient(self.obs, self.model)

        assert np.isclose(corr_coef, 0.9910875774653888)

    def test_index_of_agreement(self):
        ioa = stats.compute_index_of_agreement(self.obs, self.model)

        assert np.isclose(ioa, 0.9887577663580477)

    def test_mean_square_error(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=False)

        assert np.isclose(mse, 0.019124891068518297)

    def test_mean_square_error_centered(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=True)

        assert np.isclose(mse, 0.019110920485083147)

    def test_murphy_skill_score(self):
        mss = stats.compute_murphy_skill_score(self.obs, self.model)

        assert np.isclose(mss, 0.9617732386418576)

    def test_root_mean_square_error(self):
        rmse = stats.compute_root_mean_square_error(self.obs, self.model)

        assert np.isclose(rmse, 0.13829277301622922)

    def test_descriptive_statistics(self):
        max, min, mean, std = stats.compute_descriptive_statistics(self.model, ddof=0)

        assert np.isclose(max, 0.782928)
        assert np.isclose(min, -0.725157)
        assert np.isclose(mean, 0.031691)
        assert np.isclose(std, 0.690945)

    def test_stats(self):
        stats_output = stats.compute_stats(self.obs, self.model)

        assert isinstance(stats_output, dict)
        assert len(stats_output) == 7
