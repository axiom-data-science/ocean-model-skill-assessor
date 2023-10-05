import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_model_skill_assessor as omsa


@pytest.mark.mpl_image_compare
def test_line():
    """Test line plot with nothing extra."""

    t = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    x = np.linspace(0, 10, t.size)
    obs = pd.DataFrame({"xaxis": t, "yaxis": x**2})
    model = xr.Dataset({"xaxis": t, "yaxis": x**3})
    fig = omsa.plot.line.plot(obs, model, "xaxis", "yaxis", return_plot=True)
    return fig


# @pytest.mark.mpl_image_compare
# def test_selection():
#     # have one sample dataset that I slice different ways to select diff featuretypes
#     lon, lat, depth = -98, 30, 0
#     ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
#     # data
#     obs = pd.DataFrame(
#         {"temp": np.sin(ref_times.values.astype("float32"))}, index=ref_times
#     )
#     obs["lon"] = lon
#     obs["lat"] = lat
#     obs["depth"] = depth
#     obs.index.name = "date_time"
#     obs = obs.reset_index()

#     # model
#     # sample_times = pd.date_range(start="2000-12-", end="2001-01-04", freq="D")
#     model = xr.Dataset()
#     model["date_time"] = ("date_time", ref_times)
#     model["temp"] = ("date_time", np.sin(ref_times.values.astype("float32")))
#     model["lon"] = lon
#     model["lat"] = lat
#     model["depth"] = depth
#     # sample = pd.DataFrame(
#     #     {"FAKE_SAMPLES": np.sin(sample_times.values.astype("float32"))},
#     #     index=sample_times,
#     # )
#     featuretype = "timeSeries"
#     key_variable = "temp"
#     stats = omsa.stats.compute_stats(obs[key_variable], model[key_variable])
#     vocab_labels = {"temp": "Sea water temperature [C]"}
#     fig = omsa.plot.selection(obs, model, featuretype, key_variable, featuretype, stats,
#                         vocab_labels=vocab_labels, return_plot=True)
#     return fig

#     # line.plot(obs, model, xname="reference", yname="sample", title="test")
#     # obs: Union[DataFrame, Dataset],
#     # model: Dataset,
#     # xname: str,
#     # yname: str,
#     # title: str,
#     # xlabel: str = None,
#     # ylabel: str = None,
#     # figname: str = "figure.png",
#     # dpi: int = 100,
#     # # stats: dict = None,
#     # figsize: tuple = (15, 5),


def test_map_no_cartopy():

    CARTOPY_AVAILABLE = omsa.plot.map.CARTOPY_AVAILABLE
    omsa.plot.map.CARTOPY_AVAILABLE = False

    maps = np.array(np.ones((2, 4)))
    figname = "test"
    dsm = xr.Dataset()

    with pytest.raises(ModuleNotFoundError):
        omsa.plot.map.plot_map(maps, figname, dsm)

    omsa.plot.map.CARTOPY_AVAILABLE = CARTOPY_AVAILABLE
