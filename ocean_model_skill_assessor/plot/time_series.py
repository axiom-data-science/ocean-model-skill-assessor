"""
Time series plots.
"""

from matplotlib.pyplot import legend, subplots
from pandas import DataFrame


fs = 14
fs_title = 16
lw = 2
col_model = "r"
col_obs = "k"


def plot(
    reference: DataFrame,
    sample: DataFrame,
    title: str,
    ylabel: str = None,
    figname: str = "figure.png",
    dpi: int = 100,
    stats: dict = None,
):
    """Plot time series

    Plot reference vs. sample as time series line plot.

    Parameters
    ----------
    reference: DataFrame
        Observation time series
    sample: DataFrame
        Model time series to compare against reference.
    title: str
        Title for plot.
    ylabel: str
        Label for y-axis.
    figname: str
        Filename for figure (as absolute or relative path).
    dpi: int, optional
        dpi for figure.
    stats : dict, optional
        Statistics describing comparison, output from `df.omsa.compute_stats`.
    """
    fig, ax = subplots(1, 1, figsize=(15, 5))
    reference.plot(ax=ax, label="observation", fontsize=fs, lw=lw, color=col_obs)
    sample.plot(ax=ax, label="model", fontsize=fs, lw=lw, color=col_model)

    if stats is not None:
        stat_sum = ""
        types = ["bias", "corr", "ioa", "mse", "mss", "rmse", "dist"]
        for type in types:
            stat_sum += f"{type}: {stats[type]['value']:.1f}  "
        title = f"{title}: {stat_sum}"

    ax.set_title(title, fontsize=fs_title, loc="left")
    ax.set_xlabel("", fontsize=fs)  # don't need time label
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs)
    legend(loc="best")

    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
