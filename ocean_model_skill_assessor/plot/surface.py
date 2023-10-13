"""Surface plot."""


import pathlib

from typing import Optional, Union

import cf_pandas
import cf_xarray
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pandas import DataFrame
from xarray import Dataset

import ocean_model_skill_assessor as omsa


fs = 14
fs_title = 16


def plot(
    obs: Union[DataFrame, Dataset],
    model: Dataset,
    xname: str,
    yname: str,
    zname: str,
    suptitle: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    along_transect_distance: bool = False,
    kind="pcolormesh",
    nsubplots: int = 3,
    figname: Union[str, pathlib.Path] = "figure.png",
    dpi: int = 100,
    figsize=(15, 4),
    return_plot: bool = False,
    **kwargs,
):
    """Plot scatter or surface plot.

    For featuretype of trajectoryProfile or timeSeriesProfile.

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
    zname : str
        Name of variable to plot with color when interpreted with cf-xarray and cf-pandas
    suptitle: str, optional
        Title for plot, over all the subplots.
    xlabel: str, optional
        Label for x-axis.
    ylabel: str, optional
        Label for y-axis.
    zlabel: str, optional
        Label for colorbar.
    along_transect_distance:
        Set to True to calculate the along-transect distance in km from the longitude and latitude, which must be interpretable through cf-pandas or cf-xarray as "longitude" and "latitude".
    kind: str
        Can be "pcolormesh" for surface plot or "scatter" for scatter plot.
    nsubplots : int, optional
        Number of subplots. Might always be 3, and that is the default.
    figname: str
        Filename for figure (as absolute or relative path).
    dpi: int, optional
        dpi for figure. Default is 100.
    figsize : tuple, optional
        Figsize to pass to `plt.figure()`. Default is (15,5).
    return_plot : bool
        If True, return plot. Use for testing.
    """

    # want obs and data as DataFrames
    if kind == "scatter":
        if isinstance(obs, xr.Dataset):
            obs = obs.to_dataframe()
        if isinstance(model, xr.Dataset):
            model = model.to_dataframe().reset_index()
        # using .values on obs prevents name clashes for time and depth
        model["diff"] = obs.cf[zname].values - model.cf[zname]
    # want obs and data as Datasets
    elif kind == "pcolormesh":
        if isinstance(obs, pd.DataFrame):
            obs = obs.to_xarray()
            obs = obs.assign_coords(
                {obs.cf["T"].name: obs.cf["T"], model.cf["Z"].name: obs.cf["Z"]}
            )
        if isinstance(model, pd.DataFrame):
            model = model.to_xarray()
        # using .values on obs prevents name clashes for time and depth
        model["diff"] = obs.cf[zname].values - model.cf[zname]
        # model["diff"] = obs.cf[zname].values - model.cf[zname]
        model["diff"].attrs = {}
    else:
        raise ValueError("`kind` should be scatter or pcolormesh.")

    if along_transect_distance:
        obs["distance"] = omsa.utils.calculate_distance(
            obs.cf["longitude"], obs.cf["latitude"]
        )
        if isinstance(model, xr.Dataset):
            model["distance"] = (
                model.cf["T"].name,
                omsa.utils.calculate_distance(
                    model.cf["longitude"], model.cf["latitude"]
                ),
            )
            model = model.assign_coords({"distance": model["distance"]})
        elif isinstance(model, pd.DataFrame):
            model["distance"] = omsa.utils.calculate_distance(
                model.cf["longitude"], model.cf["latitude"]
            )

        # diff = diff.assign_coords({"distance": distance})

    # for first two plots
    # vmin, vmax, cmap, extend, levels, norm
    cmap_params = xr.plot.utils._determine_cmap_params(
        np.vstack((obs.cf[zname].values, model.cf[zname].values)), robust=True
    )
    # including `center=0` forces this to return the diverging colormap option
    cmap_params_diff = xr.plot.utils._determine_cmap_params(
        model["diff"].values, robust=True, center=0
    )

    # sharex and sharey removed the y ticklabels so don't use.
    # maybe don't work with layout="constrained"
    fig, axes = plt.subplots(
        1,
        nsubplots,
        figsize=figsize,
        layout="constrained",
    )
    #  sharex=True, sharey=True)

    # setup
    xarray_kwargs = dict(
        add_labels=False,
        add_colorbar=False,
    )
    pandas_kwargs = dict(colorbar=False)

    kwargs = {key: cmap_params.get(key) for key in ["vmin", "vmax", "cmap"]}

    if kind == "scatter":
        obs.plot(
            kind=kind,
            x=obs.cf[xname].name,
            y=obs.cf[yname].name,
            c=obs.cf[zname].name,
            ax=axes[0],
            **kwargs,
            **pandas_kwargs,
        )
    elif kind == "pcolormesh":
        obs.cf[zname].cf.plot.pcolormesh(
            x=xname, y=yname, ax=axes[0], **kwargs, **xarray_kwargs
        )
    axes[0].set_title("Observation", fontsize=fs_title)
    axes[0].set_ylabel(ylabel, fontsize=fs)
    axes[0].set_xlabel(xlabel, fontsize=fs)
    axes[0].tick_params(axis="both", labelsize=fs)

    # plot model
    if kind == "scatter":
        model.plot(
            kind=kind,
            x=model.cf[xname].name,
            y=model.cf[yname].name,
            c=model.cf[zname].name,
            ax=axes[1],
            **kwargs,
            **pandas_kwargs,
        )
    elif kind == "pcolormesh":
        model.cf[zname].cf.plot.pcolormesh(
            x=xname, y=yname, ax=axes[1], **kwargs, **xarray_kwargs
        )
    axes[1].set_title("Model", fontsize=fs_title)
    axes[1].set_xlabel(xlabel, fontsize=fs)
    axes[1].set_ylabel("")
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    # save space by not relabeling y axis
    axes[1].set_yticklabels("")
    axes[1].tick_params(axis="x", labelsize=fs)

    # plot difference (assume Dataset)
    # for last (diff) plot
    kwargs.update({key: cmap_params_diff.get(key) for key in ["vmin", "vmax", "cmap"]})
    if kind == "scatter":
        model.plot(
            kind=kind,
            x=model.cf[xname].name,
            y=model.cf[yname].name,
            c="diff",
            ax=axes[2],
            **kwargs,
            **pandas_kwargs,
        )
    elif kind == "pcolormesh":
        model["diff"].cf.plot.pcolormesh(
            x=xname, y=yname, ax=axes[2], **kwargs, **xarray_kwargs
        )
    axes[2].set_title("Obs - Model", fontsize=fs_title)
    axes[2].set_xlabel(xlabel, fontsize=fs)
    axes[2].set_ylabel("")
    axes[2].set_xlim(axes[0].get_xlim())
    axes[2].set_ylim(axes[0].get_ylim())
    axes[2].set_ylim(obs.cf[yname].min(), obs.cf[yname].max())
    axes[2].set_yticklabels("")
    axes[2].tick_params(axis="x", labelsize=fs)

    # two colorbars, 1 for obs and model and 1 for diff
    # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
    norm = mpl.colors.Normalize(vmin=cmap_params["vmin"], vmax=cmap_params["vmax"])
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_params["cmap"])
    cbar1 = fig.colorbar(mappable, ax=axes[:2], orientation="horizontal", shrink=0.5)
    cbar1.set_label(zlabel, fontsize=fs)
    cbar1.ax.tick_params(axis="both", labelsize=fs)

    norm = mpl.colors.Normalize(
        vmin=cmap_params_diff["vmin"], vmax=cmap_params_diff["vmax"]
    )
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_params_diff["cmap"])
    cbar2 = fig.colorbar(mappable, ax=axes[2], orientation="horizontal")  # shrink=0.6)
    cbar2.set_label(f"{zlabel} difference", fontsize=fs)
    cbar2.ax.tick_params(axis="both", labelsize=fs)

    fig.suptitle(suptitle, wrap=True, fontsize=fs_title)  # , loc="left")

    fig.savefig(figname, dpi=dpi)  # , bbox_inches="tight")

    if return_plot:
        return fig
