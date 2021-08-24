"""
Time series plots.
"""


import matplotlib.pyplot as plt


# matplotlib.use('agg')  # noqa


fs = 14
fs_title = 16
lw = 2


def plot(reference, sample, title, ylabel=None, figname="figure.png", dpi=100):
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
    dpi: int
        dpi for figure.

    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    reference.plot(ax=ax, label="observation", fontsize=fs, lw=lw)
    sample.plot(ax=ax, label="model", fontsize=fs, lw=lw)

    ax.set_title(title, fontsize=fs_title)
    ax.set_xlabel("", fontsize=fs)  # don't need time label
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs)
    plt.legend(loc="best")
    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
