"""
Plotting functions available for ocean-model-skill-assessor.
"""

import pathlib

from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xcmocean

from matplotlib.pyplot import figure

from . import line, map, surface


def selection(
    obs: Union[pd.DataFrame, xr.Dataset],
    model: xr.Dataset,
    featuretype: str,
    key_variable: str,
    source_name: str,
    stats: dict,
    figname: Union[str, pathlib.Path],
    vocab_labels: Optional[dict] = None,
    **kwargs,
) -> figure:
    """Plot."""

    if vocab_labels is not None:
        key_variable_label = vocab_labels[key_variable]
    else:
        key_variable_label = key_variable

    # cmap and cmapdiff selection based on key_variable name
    da = xr.DataArray(name=key_variable)

    # title
    stat_sum = ""
    types = ["bias", "corr", "ioa", "mse", "ss", "rmse"]
    if "dist" in stats:
        types += ["dist"]
    for type in types:
        # stat_sum += f"{type}: {stats[type]:.1f}  "
        stat_sum += f"{type}: {stats[type]['value']:.1f}  "

    # add location info
    # always show first/only location
    loc = f"lon: {obs.cf['longitude'][0]:.2f} lat: {obs.cf['latitude'][0]:.2f}"
    # time = f"{str(obs.cf['T'][0].date())}"  # worked for DF
    time = str(pd.Timestamp(obs.cf["T"].values[0]).date())  # works for DF and DS
    # only shows depths if 1 depth since otherwise will be on plot
    if np.unique(obs.cf["Z"]).size == 1:
        depth = f"depth: {obs.cf['Z'][0]}"
        title = f"{source_name}: {stat_sum}\n{time} {depth} {loc}"
    else:
        title = f"{source_name}: {stat_sum}\n{time} {loc}"

    # use featuretype to determine plot type
    with xr.set_options(cmap_sequential=da.cmo.seq, cmap_divergent=da.cmo.div):
        if featuretype == "timeSeries":
            xname, yname = "T", key_variable
            xlabel, ylabel = "", key_variable_label
            fig = line.plot(
                obs,
                model,
                xname,
                yname,
                title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=(15, 5),
                figname=figname,
                return_plot=True,
                **kwargs,
            )

        elif featuretype == "profile":
            xname, yname = key_variable, "Z"
            xlabel, ylabel = key_variable_label, "Depth [m]"
            fig = line.plot(
                obs,
                model,
                xname,
                yname,
                title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=(4, 8),
                figname=figname,
                return_plot=True,
                **kwargs,
            )

        elif featuretype == "trajectoryProfile":
            xname, yname, zname = "distance", "Z", key_variable
            xlabel, ylabel, zlabel = (
                "along-transect distance [km]",
                "Depth [m]",
                key_variable_label,
            )
            if "distance" not in obs.cf:
                along_transect_distance = True
            else:
                along_transect_distance = False
            fig = surface.plot(
                obs,
                model,
                xname,
                yname,
                zname,
                title,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                nsubplots=3,
                figsize=(15, 6),
                figname=figname,
                along_transect_distance=along_transect_distance,
                kind="scatter",
                return_plot=True,
                **kwargs,
            )

        elif featuretype == "timeSeriesProfile":
            xname, yname, zname = "T", "Z", key_variable
            xlabel, ylabel, zlabel = "", "Depth [m]", key_variable_label
            fig = surface.plot(
                obs,
                model,
                xname,
                yname,
                zname,
                title,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                kind="pcolormesh",
                figsize=(15, 6),
                figname=figname,
                return_plot=True,
                **kwargs,
            )

    return fig
