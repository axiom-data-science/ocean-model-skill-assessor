import numpy as np
import pandas as pd

from xarray import DataArray

from ocean_model_skill_assessor import stats


class TestStats:
    ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    obs = pd.DataFrame(
        {"obs": np.sin(ref_times.values.astype("float32"))}, index=ref_times
    )

    model_times = pd.date_range(start="2000-12-28", end="2001-01-04", freq="D")
    data = 1.25 * np.sin(model_times.values.astype("float32") + 2)
    model = pd.DataFrame({"model": data}, index=model_times)

    aligned_signals = stats._align(obs, model)
    da = DataArray(data, coords=[model_times], dims=["time"])

    aligned_signals_xr = stats._align(obs, da)

    def test_align(self):

        assert isinstance(self.aligned_signals, pd.DataFrame)
        assert self.aligned_signals.shape == (5, 2)
        assert np.isclose(self.aligned_signals["model"].mean(), -0.35140932)
        assert np.isclose(self.aligned_signals["obs"].mean(), -0.28112745)

    def test_align_xr(self):

        assert isinstance(self.aligned_signals_xr, pd.DataFrame)
        assert self.aligned_signals_xr.shape == (5, 2)
        assert np.isclose(self.aligned_signals_xr["model"].mean(), -0.35140932)
        assert np.isclose(self.aligned_signals_xr["obs"].mean(), -0.28112745)

    def test_bias(self):
        bias = stats.compute_bias(self.obs, self.model)

        assert np.isclose(bias, -0.07028185)

    def test_correlation_coefficient(self):
        corr_coef = stats.compute_correlation_coefficient(self.obs, self.model)

        assert np.isclose(corr_coef, 1.0)

    def test_index_of_agreement(self):
        ioa = stats.compute_index_of_agreement(self.obs, self.model)

        assert np.isclose(ioa, 0.9845252502709627)

    def test_mean_square_error(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=False)

        assert np.isclose(mse, 0.024080737)

    def test_mean_square_error_centered(self):
        mse = stats.compute_mean_square_error(self.obs, self.model, centered=True)

        assert np.isclose(mse, 0.019141199)

    def test_murphy_skill_score(self):
        mss = stats.compute_murphy_skill_score(self.obs, self.model)

        assert np.isclose(mss, 0.9213713780045509)

    def test_root_mean_square_error(self):
        rmse = stats.compute_root_mean_square_error(self.obs, self.model)

        assert np.isclose(rmse, 0.1551797)

    def test_descriptive_statistics(self):
        max, min, mean, std = stats.compute_descriptive_statistics(self.model, ddof=0)

        assert np.isclose(max, 0.882148)
        assert np.isclose(min, -1.247736)
        assert np.isclose(mean, -0.301843)
        assert np.isclose(std, 0.757591)

    def test_stats(self):
        stats_output = stats.compute_stats(self.obs, self.model)

        assert isinstance(stats_output, dict)
        assert len(stats_output) == 7
