import cf_pandas
import extract_model as em
import numpy as np
import pandas as pd
import xarray as xr

from ocean_model_skill_assessor import stats


class TestStats:
    ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    obs = pd.DataFrame(
        {"temp": np.sin(ref_times.values.astype("float32"))}, index=ref_times
    )
    obs.index.name = "time"

    model_times = pd.date_range(start="2000-12-28", end="2001-01-04", freq="D")
    data = 1.25 * np.sin(model_times.values.astype("float32") + 2)
    # model = pd.DataFrame({"model": data}, index=model_times)
    # model = xr.DataArray(data, coords=[model_times], dims=["time"])
    # model.index.name = "date_time"
    model = xr.Dataset()
    model["time"] = model_times
    model["time"].attrs["axis"] = "T"
    model["temp"] = ("time", data)
    # use em.select here to align
    model = em.select(model, T=obs.cf["T"])

    # aligned_signals = stats._align(obs, model)
    # da = xr.DataArray(data, coords=[model_times], dims=["time"])
    # da["time"].attrs = {"axis": "T"}
    # aligned_signals_xr = stats._align(obs, da)

    def test_select(self):
        assert isinstance(self.obs, pd.DataFrame)
        assert isinstance(self.model, xr.Dataset)
        assert self.model.to_array().shape == (1, 17)
        assert np.isclose(self.model.to_array().mean(), -0.31737685)
        assert np.isclose(self.obs.mean(), -0.08675907)

    # def test_align_xr(self):
    #     assert isinstance(self.aligned_signals_xr, pd.DataFrame)
    #     assert self.aligned_signals_xr.shape == (17, 2)
    #     assert np.isclose(self.aligned_signals_xr["model"].mean(), -0.31737685)
    #     assert np.isclose(self.aligned_signals_xr["obs"].mean(), -0.08675907)

    def test_bias(self):
        bias = stats.compute_bias(self.obs["temp"], self.model["temp"])

        assert np.isclose(bias, -0.23061779141426086)

    def test_correlation_coefficient(self):
        corr_coef = stats.compute_correlation_coefficient(
            self.obs["temp"], self.model["temp"]
        )

        assert np.isclose(corr_coef, 0.906813)

    def test_index_of_agreement(self):
        ioa = stats.compute_index_of_agreement(self.obs["temp"], self.model["temp"])

        assert np.isclose(ioa, 0.9174428656697273)

    def test_mean_square_error(self):
        mse = stats.compute_mean_square_error(
            self.obs["temp"], self.model["temp"], centered=False
        )

        assert np.isclose(mse, 0.14343716204166412)

    def test_mean_square_error_centered(self):
        mse = stats.compute_mean_square_error(
            self.obs["temp"], self.model["temp"], centered=True
        )

        assert np.isclose(mse, 0.0902525931596756)

    def test_murphy_skill_score(self):
        mss = stats.compute_murphy_skill_score(self.obs["temp"], self.model["temp"])

        assert np.isclose(mss, 0.7155986726284027)

    def test_root_mean_square_error(self):
        rmse = stats.compute_root_mean_square_error(
            self.obs["temp"], self.model["temp"]
        )

        assert np.isclose(rmse, 0.3787309890168272)

    def test_descriptive_statistics(self):
        max, min, mean, std = stats.compute_descriptive_statistics(
            self.model["temp"], ddof=0
        )

        assert np.isclose(max, 0.882148)
        assert np.isclose(min, -1.2418900728225708)
        assert np.isclose(mean, -0.31737685378860025)
        assert np.isclose(std, 0.6187897906117683)

    def test_stats(self):
        stats_output = stats.compute_stats(self.obs["temp"], self.model["temp"])

        assert isinstance(stats_output, dict)
        assert len(stats_output) == 7
