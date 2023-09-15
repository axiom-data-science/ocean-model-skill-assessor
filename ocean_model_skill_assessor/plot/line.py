"""
Time series plots.
"""

# from matplotlib.pyplot import legend, subplots
from typing import Union

import matplotlib.pyplot as plt

from pandas import DataFrame


fs = 14
fs_title = 16
lw = 2
col_model = "r"
col_obs = "k"


def plot(
    df: DataFrame,
    xname: Union[str, list],
    yname: Union[str, list],
    # reference: DataFrame,
    # sample: DataFrame,
    title: str,
    xlabel: str = None,
    ylabel: str = None,
    figname: str = "figure.png",
    dpi: int = 100,
    stats: dict = None,
    figsize: tuple = (15, 5),
    **kwargs,
):
    """Plot time series or CTD profile.

    Plot reference vs. sample as time series line plot.

    Parameters
    ----------
    reference: DataFrame
        Observation time series
    sample: DataFrame
        Model time series to compare against reference.
    title: str
        Title for plot.
    xlabel: str
        Label for x-axis.
    ylabel: str
        Label for y-axis.
    figname: str
        Filename for figure (as absolute or relative path).
    dpi: int, optional
        dpi for figure.
    stats : dict, optional
        Statistics describing comparison, output from `df.omsa.compute_stats`.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # probably CTD profile plot (depth on y axis)
    if isinstance(xname, list):
        for name in xname:
            if name == "obs":
                label = "data"
                color = col_obs
            elif name == "model":
                label = "model"
                color = col_model
            df.plot(
                x=name,
                y=yname,
                ax=ax,
                fontsize=fs,
                lw=lw,
                subplots=False,
                label=label,
                color=color,
            )
    # probably time series plot
    elif isinstance(yname, list):
        for name in yname:
            if name == "obs":
                label = "data"
                color = col_obs
            elif name == "model":
                label = "model"
                color = col_model
            df.plot(
                x=xname,
                y=name,
                ax=ax,
                fontsize=fs,
                lw=lw,
                subplots=False,
                label=label,
                color=color,
            )
        ax.set_xlim(df[xname].min(), df[xname].max())
    # df[xname].plot(ax=ax, label="observation", fontsize=fs, lw=lw, color=col_obs)
    # df[yname].plot(ax=ax, label="model", fontsize=fs, lw=lw, color=col_model)
    if stats is not None:
        stat_sum = ""
        types = ["bias", "corr", "ioa", "mse", "ss", "rmse"]
        if "dist" in stats:
            types += ["dist"]
        for type in types:
            stat_sum += f"{type}: {stats[type]['value']:.1f}  "
            # add line mid title if tall plot instead of wide plot
            if type == "ioa" and figsize[1] > figsize[0]:
                stat_sum += "\n"

        title = f"{title}: {stat_sum}"

    ax.set_title(title, fontsize=fs_title, loc="left")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs)
    plt.legend(loc="best")

    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
