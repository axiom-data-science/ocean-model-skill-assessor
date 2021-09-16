import os

import numpy as np
import pandas as pd

from xarray import DataArray

from ocean_model_skill_assessor import stats


class TestStats:
    ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    obs = pd.DataFrame({"obs": np.sin(ref_times.view(int))}, index=ref_times)

    model_times = pd.date_range(start="2000-12-28", end="2001-01-04", freq="D")
    data = 1.25 * np.sin(model_times.view(int) + 2)
    model = pd.DataFrame({"model": data}, index=model_times)

    aligned_signals = stats._align(obs, model)
    da = DataArray(data, coords=[model_times], dims=["time"])

    aligned_signals_xr = stats._align(obs, da)

    def test_align(self):

        assert isinstance(self.aligned_signals, pd.DataFrame)
        assert self.aligned_signals.shape == (5, 2)
        assert np.isclose(self.aligned_signals["model"].mean(), 0.16391766802943322)
        assert np.isclose(self.aligned_signals["obs"].mean(), 0.13113413442354657)

    def test_align_xr(self):

        assert isinstance(self.aligned_signals_xr, pd.DataFrame)
        assert self.aligned_signals_xr.shape == (5, 2)
        assert np.isclose(self.aligned_signals_xr["model"].mean(), 0.16391766802943322)
        assert np.isclose(self.aligned_signals_xr["obs"].mean(), 0.13113413442354657)

    def test_bias(self):
        bias = stats.compute_bias(self.obs, self.model)

        assert np.isclose(bias, 0.032783533605886636)

    def test_correlation_coefficient(self):
        corr_coef = stats.compute_correlation_coefficient(self.obs, self.model)

        assert np.isclose(corr_coef, 1.0)

    def test_index_of_agreement(self):
        ioa = stats.compute_index_of_agreement(self.obs, self.model)

        assert np.isclose(ioa, 0.9872247131609245)

    def test_mean_square_error(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=False)

        assert np.isclose(mse, 0.03156566684130121)

    def test_mean_square_error_centered(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=True)

        assert np.isclose(mse, 0.03049090676561291)

    def test_murphy_skill_score(self):
        mss = stats.compute_murphy_skill_score(self.obs, self.model)

        assert np.isclose(mss, 0.935296965985732)

    def test_root_mean_square_error(self):
        rmse = stats.compute_root_mean_square_error(self.obs, self.model)

        assert np.isclose(rmse, 0.17766729254790034)

    def test_descriptive_statistics(self):
        max, min, mean, std = stats.compute_descriptive_statistics(self.model, ddof=0)

        assert np.isclose(max, 0.97866)
        assert np.isclose(min, -0.906447)
        assert np.isclose(mean, 0.039614)
        assert np.isclose(std, 0.863681)

    def test_stats(self):
        stats_output = stats.compute_stats(self.obs, self.model)

        assert isinstance(stats_output, dict)
        assert len(stats_output) == 7


class TestStatsSummary:
    ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    obs = pd.DataFrame({"obs": np.sin(ref_times.view(int))}, index=ref_times)

    model_times = pd.date_range(start="2000-12-28", end="2001-01-04", freq="D")
    data = 1.25 * np.sin(model_times.view(int) + 2)
    model = pd.DataFrame({"model": data}, index=model_times)

    aligned_signals = stats._align(obs, model)
    stats_output = stats.compute_stats(obs, model)

    def test_stats_summary(self):
        stats_summary = stats.StatsSummary('fake-approach', 'fake-loc')
        stats_summary.add_dataset('fake-dataset-id', 'fake-variable', self.stats_output)

        assert stats_summary.description == 'fake-approach_fake-loc'
        assert list(stats_summary.stats.columns.values) == [
            'dataset_id',
            'variable',
            'max',
            'min',
            'mean',
            'std',
            'bias',
            'corr',
            'ioa',
            'mse',
            'mss',
            'rmse'
        ]
        assert stats_summary.stats.dataset_id.values == ['fake-dataset-id']
        assert stats_summary.stats.variable.values == ['fake-variable']
        assert np.isclose(stats_summary.stats['max'].values, 0.97866)
        assert np.isclose(stats_summary.stats['min'].values, -0.906447)
        assert np.isclose(stats_summary.stats['mean'].values, 0.039614)
        assert np.isclose(stats_summary.stats['std'].values, 0.863681)
        assert np.isclose(stats_summary.stats['bias'].values, 0.032783533605886636)
        assert np.isclose(stats_summary.stats['corr'].values, 1.0)
        assert np.isclose(stats_summary.stats['ioa'].values, 0.9872247131609245)
        assert np.isclose(stats_summary.stats['mse'].values, 0.03156566684130121)
        assert np.isclose(stats_summary.stats['mss'].values, 0.935296965985732)
        assert np.isclose(stats_summary.stats['rmse'].values, 0.17766729254790034)

    def test_stats_summary_csv(self):
        stats_summary = stats.StatsSummary('fake-approach', 'fake-loc')
        stats_summary.add_dataset('fake-dataset-id', 'fake-variable', self.stats_output)
        try:
            stats_summary.to_csv('fake-stats.csv')
            loaded_stats = pd.read_csv('fake-stats.csv')
            assert pd.testing.assert_frame_equal(stats_summary.stats, loaded_stats) is None
        finally:
            os.remove('fake-stats.csv')
