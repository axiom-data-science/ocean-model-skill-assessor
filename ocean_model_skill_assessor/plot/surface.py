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
    model_title: str = "Model",
    along_transect_distance: bool = False,
    plot_on_map: bool = False,
    proj=None,
    extent=None,
    kind="pcolormesh",
    nsubplots: int = 3,
    figname: Union[str, pathlib.Path] = "figure.png",
    dpi: int = 100,
    figsize=(15, 6),
    return_plot: bool = False,
    invert_yaxis: bool = False,
    make_Z_negative=None,
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

    if "override_plot" in kwargs:
        kwargs.pop("override_plot")

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
    if plot_on_map:
        if proj is None:
            import cartopy

            proj = cartopy.crs.Mercator()
        subplot_kw = dict(projection=proj, frameon=False)
    else:
        subplot_kw = {}

    if make_Z_negative is not None:
        if make_Z_negative == "obs":
            if (obs[obs.cf["Z"].notnull()].cf["Z"] > 0).all():
                obs[obs.cf["Z"].name] = -obs.cf["Z"]
        elif make_Z_negative == "model":
            if (model[model.cf["Z"].notnull()].cf["Z"] > 0).all():
                model[model.cf["Z"].name] = -model.cf["Z"]

    fig, axes = plt.subplots(
        1,
        nsubplots,
        figsize=figsize,
        layout="constrained",
        subplot_kw=subplot_kw,
    )
    #  sharex=True, sharey=True)

    # setup
    xarray_kwargs = dict(
        add_labels=False,
        add_colorbar=False,
    )
    pandas_kwargs = dict(colorbar=False)

    kwargs.update({key: cmap_params.get(key) for key in ["vmin", "vmax", "cmap"]})

    if plot_on_map:
        omsa.plot.map.setup_ax(
            axes[0], left_labels=True, bottom_labels=True, top_labels=False, fontsize=12
        )
        kwargs["transform"] = omsa.plot.map.pc
        if extent is not None:
            axes[0].set_extent(extent)
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
    if invert_yaxis:
        axes[0].invert_yaxis()

    # plot model
    if plot_on_map:
        omsa.plot.map.setup_ax(
            axes[1],
            left_labels=False,
            bottom_labels=True,
            top_labels=False,
            fontsize=12,
        )
        if extent is not None:
            axes[1].set_extent(extent)
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
    axes[1].set_title(model_title, fontsize=fs_title)
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
    if plot_on_map:
        omsa.plot.map.setup_ax(
            axes[2],
            left_labels=False,
            bottom_labels=True,
            top_labels=False,
            fontsize=12,
        )
        if extent is not None:
            axes[2].set_extent(extent)
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
    # CAN SEE 3 PLOTS
    axes[2].set_title("Obs - Model", fontsize=fs_title)
    axes[2].set_xlabel(xlabel, fontsize=fs)
    axes[2].set_ylabel("")
    if not plot_on_map:
        axes[2].set_xlim(axes[0].get_xlim())
        axes[2].set_ylim(axes[0].get_ylim())
        axes[2].set_ylim(obs.cf[yname].min(), obs.cf[yname].max())
        axes[2].set_yticklabels("")
        axes[2].tick_params(axis="x", labelsize=fs)
    # import pdb; pdb.set_trace()

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
