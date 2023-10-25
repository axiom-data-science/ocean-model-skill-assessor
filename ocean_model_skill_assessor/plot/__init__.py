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

from . import line, map, quiver, surface


def selection(
    obs: Union[pd.DataFrame, xr.Dataset],
    model: xr.Dataset,
    featuretype: str,
    key_variable: str,
    source_name: str,
    stats: dict,
    figname: Union[str, pathlib.Path],
    vocab_labels: Optional[dict] = None,
    xcmocean_options: Optional[dict] = None,
    **kwargs,
) -> figure:
    """Plot."""

    # cmap and cmapdiff selection based on key_variable name
    # key_variable is always a list now (though might only have 1 entry total)
    # in any case the variables should be related and use the same colormap
    da = xr.DataArray(name=key_variable[0])

    # must contain keys
    if xcmocean_options is not None:
        if any(
            [
                key
                for key in xcmocean_options.keys()
                if key not in ["regexin", "seqin", "divin"]
            ]
        ):
            raise KeyError(
                'keys for `xcmocean_options` must be ["regexin", "seqin", "divin"]'
            )
        xcmocean.set_options(**xcmocean_options)
        context = dict(cmap_sequential=da.cmo.seq, cmap_divergent=da.cmo.div)
    else:
        context = dict(cmap_sequential=da.cmo.seq, cmap_divergent=da.cmo.div)
        try:
            assert len(context) > 0
        except AssertionError:
            context = {}

    if vocab_labels is not None:
        key_variable_label = [vocab_labels[key] for key in key_variable]
        # key_variable_label = vocab_labels[key_variable]
    else:
        key_variable_label = key_variable
    
    # back to single strings from list if only one entry
    if len(key_variable_label) == 1:
        key_variable_label = key_variable_label[0]
        key_variable = key_variable[0]

    # title
    def stats_string(stats):
        types = ["bias", "corr", "ioa", "mse", "ss", "rmse"]
        if "dist" in stats:
            types += ["dist"]
        stat_sum_sub = "".join([f"{type}: {stats[type]['value']:.1f}  " for type in types])
        # for type in types:
        #     # stat_sum += f"{type}: {stats[type]:.1f}  "
        #     stat_sum = f"{type}: {stats[type]['value']:.1f}  "
        return stat_sum_sub
   
    
    if isinstance(stats, list):
        stat_sum = ""
        for stat, key in zip(stats, key_variable):
            stat_sum += f"{key}: "
            stat_sum += stats_string(stat)
        
    elif isinstance(stats, dict):
        stat_sum = stats_string(stats)
        
    # add location info
    # always show first/only location
    if obs.cf["longitude"].size == 1:
        loc = f"lon: {float(obs.cf['longitude']):.2f} lat: {float(obs.cf['latitude']):.2f}"
    elif isinstance(obs, pd.DataFrame) and obs.cf["longitude"].size > 1:
        loc = f"lon: {obs.cf['longitude'][0]:.2f} lat: {obs.cf['latitude'][0]:.2f}"
    elif isinstance(obs, xr.Dataset) and obs.cf["longitude"].ndim == 1:  # untested
        loc = f"lon: {obs.cf['longitude'][0]:.2f} lat: {obs.cf['latitude'][0]:.2f}"
    elif isinstance(obs, xr.Dataset) and obs.cf["longitude"].ndim == 2:
        # locations will be plotted in this case
        loc = ""
        # loc = f"lon: {obs.cf['longitude'][0][0]:.2f} lat: {obs.cf['latitude'][0][0]:.2f}"
    # time = f"{str(obs.cf['T'][0].date())}"  # worked for DF
    time = str(pd.Timestamp(obs.cf["T"].values[0]).date())  # works for DF and DS
    
    # build title
    title = f"{source_name}: {stat_sum}\n"
    
    # don't show time in title if grid because will be putting it in each time there
    if featuretype != "grid":
        title += f"{time} "
    
    # only shows depths if 1 depth since otherwise will be on plot
    if obs.cf["Z"].size == 1:
        depth = f"depth: {obs.cf['Z'].values}"
        # title = f"{source_name}: {stat_sum}\n{time} {depth} {loc}"        
    elif np.unique(obs.cf["Z"][~np.isnan(obs.cf["Z"])]).size == 1:
        # if (np.unique(obs.cf["Z"]) * ~np.isnan(obs.cf["Z"])).size == 1:
        # if np.unique(obs[obs.cf["Z"].notnull()].cf["Z"]).size == 1:  # did not work for timeSeriesProfile
        depth = f"depth: {obs.cf['Z'][0]}"
        # title = f"{source_name}: {stat_sum}\n{time} {depth} {loc}"
    else:
        depth = None
        # title = f"{source_name}: {stat_sum}\n{time} {loc}"

    if depth is not None:
        title += f"{depth} "

    title += f"{loc}"

    # use featuretype to determine plot type
    with xr.set_options(**context):
    # with xr.set_options(cmap_sequential=da.cmo.seq, cmap_divergent=da.cmo.div):
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
            # Assume want along-transect distance if number of unique locations is
            # equal to or more than number of times
            if (
                np.unique(obs.cf["longitude"]).size >= np.unique(obs.cf["T"]).size
                or np.unique(obs.cf["latitude"]).size >= np.unique(obs.cf["T"]).size
            ):
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
            # otherwise use time for x axis
            else:
                xname, yname, zname = "T", "Z", key_variable
                xlabel, ylabel, zlabel = (
                    "",
                    "Depth [m]",
                    key_variable_label,
                )
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
                obs.squeeze(),
                model.squeeze(),
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

        elif featuretype == "grid":
            # if "figsize" in kwargs:
            #     figsize = kwargs["figsize"]
            #     kwargs.pop("figsize")
            # else:
            #     figsize = (16,6)
            # if "legend_arrow_length" in kwargs:
            #     legend_arrow_length = kwargs["legend_arrow_length"]
            #     kwargs.pop("legend_arrow_length")
            # else:
            #     legend_arrow_length = 5
            # for a vector input, do quiver plot
            if len(key_variable) == 2:
                xname, yname, uname, vname = "longitude", "latitude", key_variable[0], key_variable[1]
                xlabel, ylabel, ulabel, vlabel = "", "", key_variable_label[0], key_variable_label[1]
                # import pdb; pdb.set_trace()
                fig = quiver.plot(
                    obs.squeeze(),
                    model.squeeze(),
                    xname,
                    yname,
                    uname,
                    vname,
                    # legend_arrow_length,
                    title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ulabel=ulabel,
                    vlabel=vlabel,
                    figname=figname,
                    return_plot=True,
                    # kwargs_for_plot,
                    **kwargs,
                )
                
            # scalar surface plot
            else:
                xname, yname, zname = "T", "Z", key_variable
                xlabel, ylabel, zlabel = "", "Depth [m]", key_variable_label
                fig = surface.plot(
                    obs.squeeze(),
                    model.squeeze(),
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
