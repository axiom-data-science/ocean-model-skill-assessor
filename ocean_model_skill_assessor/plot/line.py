"""
Time series plots.
"""

import pathlib

from typing import Optional, Union

import cf_pandas
import cf_xarray
import matplotlib.pyplot as plt
import numpy as np

from pandas import DataFrame
from xarray import Dataset


fs = 14
fs_title = 16
lw = 2
col_model = "r"
col_obs = "k"


def plot(
    obs: Union[DataFrame, Dataset],
    model: Dataset,
    xname: str,
    yname: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    model_label: str = "Model",
    figname: Union[str, pathlib.Path] = "figure.png",
    dpi: int = 100,
    figsize: tuple = (15, 5),
    return_plot: bool = False,
    **kwargs,
):
    """Plot time series or CTD profile.

    Use for featuretype of timeSeries or profile.
    Plot obs vs. model as time series line plot or CTD profile.

    Parameters
    ----------
    obs: DataFrame, Dataset
        Observation time series
    mode: Dataset
        Model time series to compare against obs
    xname : str
        Name of variable to plot on x-axis when interpreted with cf-xarray and cf-pandas
    yname : str
        Name of variable to plot on y-axis when interpreted with cf-xarray and cf-pandas
    title: str, optional
        Title for plot.
    xlabel: str, optional
        Label for x-axis.
    ylabel: str, optional
        Label for y-axis.
    figname: str
        Filename for figure (as absolute or relative path).
    dpi: int, optional
        dpi for figure. Default is 100.
    figsize : tuple, optional
        Figsize to pass to `plt.figure()`. Default is (15,5).
    return_plot : bool
        If True, return plot. Use for testing.
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    ax.plot(obs.cf[xname], obs.cf[yname], label="data", lw=lw, color=col_obs)
    ax.plot(
        np.array(model.cf[xname].squeeze()),
        np.array(model.cf[yname].squeeze()),
        label=model_label,
        lw=lw,
        color=col_model,
    )

    plt.tick_params(axis="both", labelsize=fs)

    ax.set_title(title, fontsize=fs_title, loc="left", wrap=True)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs)
    plt.legend(loc="best")

    fig.savefig(
        figname,
        dpi=dpi,
    )  # , bbox_inches="tight")

    if return_plot:
        return fig
