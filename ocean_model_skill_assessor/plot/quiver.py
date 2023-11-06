"""Quiver plot."""


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


def plot_1(
    obs,
    model,
    suptitle,
    nsubplots,
    figsize,
    proj,
    indexer,
    xname,
    yname,
    uname,
    vname,
    model_title,
    scale,
    legend_arrow_length,
    extent,
    xlabel,
    ylabel,
    figname,
    dpi,
    **kwargs,
):
    """Plot 1 time/only time."""

    # sharex and sharey removed the y ticklabels so don't use.
    # maybe don't work with layout="constrained"
    fig, axes = plt.subplots(
        1,
        nsubplots,
        figsize=figsize,
        layout="constrained",
        subplot_kw=dict(projection=proj, frameon=False),
    )
    #  sharex=True, sharey=True)
    omsa.plot.map.setup_ax(
        axes[0], left_labels=True, bottom_labels=True, top_labels=False, fontsize=12
    )
    obs_plot = obs.cf.isel(indexer).plot.quiver(
        x=obs.cf[xname].name,
        y=obs.cf[yname].name,
        u=obs.cf[uname].name,
        v=obs.cf[vname].name,
        ax=axes[0],
        add_guide=False,
        angles="xy",
        scale_units="xy",
        scale=scale,
        transform=omsa.plot.map.pc,
        **kwargs,
    )
    qv_key = axes[0].quiverkey(
        obs_plot,
        0.94,
        1.03,
        legend_arrow_length,
        f"{legend_arrow_length} m/s",
        labelpos="N",
        labelsep=0.05,
        color="k",
        fontproperties=dict(size=12),
        #    transform=omsa.plot.map.pc,
    )
    if extent is not None:
        axes[0].set_extent(extent)

    axes[0].set_title("Observation", fontsize=fs_title)
    axes[0].set_ylabel(ylabel, fontsize=fs)
    axes[0].set_xlabel(xlabel, fontsize=fs)
    axes[0].tick_params(axis="both", labelsize=fs)

    # plot model
    omsa.plot.map.setup_ax(
        axes[1], left_labels=False, bottom_labels=True, top_labels=False, fontsize=12
    )
    # import pdb; pdb.set_trace()_loop
    model.cf.isel(indexer).plot.quiver(
        x=model.cf[xname].name,
        y=model.cf[yname].name,
        u=model.cf[uname].name,
        v=model.cf[vname].name,
        ax=axes[1],
        add_guide=False,
        angles="xy",
        scale_units="xy",
        scale=scale,
        transform=omsa.plot.map.pc,
        **kwargs,
    )
    if extent is not None:
        axes[1].set_extent(extent)

    axes[1].set_title(model_title, fontsize=fs_title)
    axes[1].set_xlabel(xlabel, fontsize=fs)
    axes[1].set_ylabel("")
    # axes[1].set_xlim(axes[0].get_xlim())
    # axes[1].set_ylim(axes[0].get_ylim())
    # save space by not relabeling y axis
    axes[1].set_yticklabels("")
    axes[1].tick_params(axis="x", labelsize=fs)

    # plot difference (assume Dataset)
    # model = model.rename({model.cf[uname].name: obs.cf[uname].name,
    #                       model.cf[vname].name: obs.cf[vname].name,})
    # diff = obs - model
    # subtract the variable as arrays to avoid variable name issues
    diff = obs.copy(deep=True)
    diff[diff.cf[uname].name] -= model.cf[uname].values
    diff[diff.cf[vname].name] -= model.cf[vname].values
    omsa.plot.map.setup_ax(
        axes[2], left_labels=False, bottom_labels=True, top_labels=False, fontsize=12
    )
    diff.cf.isel(indexer).plot.quiver(
        x=obs.cf[xname].name,
        y=obs.cf[yname].name,
        u=obs.cf[uname].name,
        v=obs.cf[vname].name,
        ax=axes[2],
        add_guide=False,
        angles="xy",
        scale_units="xy",
        scale=scale,
        transform=omsa.plot.map.pc,
        **kwargs,
    )
    if extent is not None:
        axes[2].set_extent(extent)

    axes[2].set_title("Obs - Model", fontsize=fs_title)
    axes[2].set_xlabel(xlabel, fontsize=fs)
    axes[2].set_ylabel("")
    # axes[2].set_xlim(axes[0].get_xlim())
    # axes[2].set_ylim(axes[0].get_ylim())
    # axes[2].set_ylim(obs.cf[yname].min(), obs.cf[yname].max())
    axes[2].set_yticklabels("")
    axes[2].tick_params(axis="x", labelsize=fs)

    fig.suptitle(suptitle, wrap=True, fontsize=fs_title)  # , loc="left")
    fig.savefig(str(figname), dpi=dpi)  # , bbox_inches="tight")

    return fig


def plot(
    obs: Dataset,
    model: Dataset,
    xname: str,
    yname: str,
    uname: str,
    vname: str,
    suptitle: str,
    figsize=(16, 6),
    legend_arrow_length: int = 5,
    scale=1,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ulabel: Optional[str] = None,
    vlabel: Optional[str] = None,
    model_title: str = "Model",
    indexer=None,
    subplot_description: str = "",
    nsubplots: int = 3,
    figname: Union[str, pathlib.Path] = "figure.png",
    dpi: int = 100,
    return_plot: bool = False,
    proj=None,
    extent=None,
    override_plot: bool = False,
    make_movie: bool = False,
    **kwargs,
):
    """Plot quiver of vectors in time.

    Times must already match between obs and model.

    If you want to change the scale, input "scale=int" as a kwarg
    UPDATE ALL OF THIS

    For featuretype of trajectoryProfile or timeSeriesProfile.

    Parameters
    ----------
    obs: Dataset
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
    legend_arrow_length : int
        Length of legend arrow in m/s or whatever units of u and v.
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

    assert isinstance(obs, xr.Dataset)

    if proj is None:
        import cartopy

        proj = cartopy.crs.Mercator()
        # proj = cartopy.crs.Mercator(central_longitude=float(central_longitude))

    if obs.cf["T"].shape == ():
        fig = plot_1(
            obs,
            model,
            suptitle,
            nsubplots,
            figsize,
            proj,
            indexer,
            xname,
            yname,
            uname,
            vname,
            model_title,
            scale,
            legend_arrow_length,
            extent,
            xlabel,
            ylabel,
            figname,
            dpi,
            **kwargs,
        )
    else:
        for ind, t in enumerate(obs.cf["T"]):
            t = str(pd.Timestamp(t.values).isoformat()[:13])
            # t = str(pd.Timestamp(t.values).date())

            # add time to title
            suptitle_use = f"{suptitle}\n{t}: {subplot_description}"

            if isinstance(figname, pathlib.Path):
                figname_loop = figname.parent / f"{figname.stem}_{t}{figname.suffix}"
            else:
                raise NotImplementedError("Need to implement for string figname")

            if figname_loop.is_file() and not override_plot:
                continue

            fig = plot_1(
                obs.cf.sel(T=t),
                model.cf.sel(T=t),
                suptitle_use,
                nsubplots,
                figsize,
                proj,
                indexer,
                xname,
                yname,
                uname,
                vname,
                model_title,
                scale,
                legend_arrow_length,
                extent,
                xlabel,
                ylabel,
                figname_loop,
                dpi,
                **kwargs,
            )

            # don't close if it is the last plot of the loop so we have something to return
            if ind != (obs.cf["T"].size - 1):
                plt.close(fig)

        if make_movie:
            import shlex
            import subprocess

            if isinstance(figname, pathlib.Path):
                comm = f"ffmpeg -r 4 -pattern_type glob -i '{figname.parent / figname.stem}_????-*.png' -c:v libx264 -pix_fmt yuv420p  -crf 25 {figname.parent / figname.stem}.mp4"
                # comm = f"ffmpeg -r 4 -pattern_type glob -i '/Users/kthyng/Library/Caches/ocean-model-skill-assessor/hfradar_ciofs/out/hfradar_lower-ci_system-B_2006-2007_all_east_north_remove-under-50-percent-data_units-to-meters_*.png' -c:v libx264 -pix_fmt yuv420p  -crf 15 out.mp4"
                subprocess.run(shlex.split(comm))
            else:
                raise NotImplementedError

    if return_plot and override_plot:
        return fig
