"""Surface plot."""


from typing import Optional, Union

import cf_pandas
import cf_xarray
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import ocean_model_skill_assessor as omsa


def plot(
    dd: Union[pd.DataFrame, xr.Dataset],
    xname: str,
    yname: str,
    zname: Union[str, list],
    nsubplots: int = 3,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figname: str = "figure.png",
    dpi: int = 100,
    stats: dict = None,
    clabel: Optional[str] = None,
    kind="pcolormesh",
    **kwargs,
):
    """Surface plot."""

    # cmap = cmap or xr.get_options()["cmap_divergent"]
    # cmap_diff = cm

    # dds = dd[zname[1]] - dd[zname[0]]
    dd["diff"] = dd[zname[1]] - dd[zname[0]]

    # # for diverging property
    # # import pdb; pdb.set_trace()
    # vmax = dd[zname + ["diff"]].max().max()
    # vmin = -vmax

    # for first two plots
    # vmin, vmax, cmap, extend, levels, norm
    cmap_params = xr.plot.utils._determine_cmap_params(dd[zname].values, robust=True)
    # including `center=0` forces this to return the diverging colormap option
    cmap_params_diff = xr.plot.utils._determine_cmap_params(
        dd["diff"].values, robust=True, center=0
    )

    # reference = pd.DataFrame(reference)
    # sample = pd.DataFrame(sample)
    if xname == "distance":
        dd["distance [km]"] = omsa.utils.calculate_distance(
            dd.cf["longitude"], dd.cf["latitude"]
        )

    fig, axes = plt.subplots(1, nsubplots, figsize=(15, 4))

    # vmax = max((dd[zname[0]].max(), dd[zname[1]].max(), dds.max(), xr.apply_ufunc(np.absolute, dd[zname[0]]).max(),
    #             xr.apply_ufunc(np.abs, dd[zname[1]]).max(), xr.apply_ufunc(np.absolute, dds).max()))
    # vmin = -vmax

    # kwargs = dict(cmap=cmap, x=dd.cf[xname].name, y=dd.cf[yname].name,
    #               vmin=vmin, vmax=vmax)

    kwargs = dict(x=dd.cf[xname].name, y=dd.cf[yname].name)
    kwargs.update({key: cmap_params.get(key) for key in ["vmin", "vmax", "cmap"]})

    xarray_kwargs = dict(add_labels=False, add_colorbar=False)
    pandas_kwargs = dict(colorbar=False)

    if isinstance(dd, xr.Dataset):
        kwargs.update(xarray_kwargs)
    elif isinstance(dd, pd.DataFrame):
        kwargs.update(pandas_kwargs)

    # plot obs
    dd.plot(kind=kind, c=zname[0], ax=axes[0], **kwargs)
    axes[0].set_title("Observation")
    axes[0].set_ylabel(kwargs["y"])
    # don't add label if x dim is time since its obvious then
    if not xname == "T":
        axes[0].set_xlabel(kwargs["x"])

    # plot model
    # import pdb; pdb.set_trace()
    dd.plot(kind=kind, c=zname[1], ax=axes[1], ylabel="", **kwargs)
    axes[1].set_title("Model")
    # don't add label if x dim is time since its obvious then
    if not xname == "T":
        axes[1].set_xlabel(kwargs["x"])
    axes[1].set_yticklabels("")

    # plot difference

    # for last (diff) plot
    kwargs.update({key: cmap_params_diff.get(key) for key in ["vmin", "vmax", "cmap"]})

    # import pdb; pdb.set_trace()
    dd.plot(
        kind=kind, c="diff", ax=axes[2], ylabel="", **kwargs
    )  # , cbar_kwargs={'label':clabel})
    axes[2].set_title("Obs - Model")
    # don't add label if x dim is time since its obvious then
    if not xname == "T":
        axes[2].set_xlabel(kwargs["x"])
    axes[2].set_yticklabels("")

    # separate colorbar(s)
    # one long colorbar
    if cmap_params["cmap"].name == cmap_params_diff["cmap"].name:
        cbar_ax = fig.add_axes([0.2, -0.1, 0.6, 0.05])  # Left, bottom, width, height.
        # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
        norm = mpl.colors.Normalize(vmin=cmap_params["vmin"], vmax=cmap_params["vmax"])
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_params["cmap"])
        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(clabel)
    # axes[2].clabel

    # two colorbars, 1 for obs and model and 1 for diff
    else:
        cbar_ax1 = fig.add_axes(
            [0.175, -0.1, 0.4, 0.05]
        )  # Left, bottom, width, height.
        # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
        norm = mpl.colors.Normalize(vmin=cmap_params["vmin"], vmax=cmap_params["vmax"])
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_params["cmap"])
        cbar1 = fig.colorbar(mappable, cax=cbar_ax1, orientation="horizontal")
        cbar1.set_label(clabel)

        cbar_ax2 = fig.add_axes(
            [0.719, -0.1, 0.15, 0.05]
        )  # Left, bottom, width, height.
        # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
        norm = mpl.colors.Normalize(
            vmin=cmap_params_diff["vmin"], vmax=cmap_params_diff["vmax"]
        )
        mappable_diff = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_params_diff["cmap"])
        cbar2 = fig.colorbar(mappable_diff, cax=cbar_ax2, orientation="horizontal")
        cbar2.set_label(f"{clabel} difference")

    # add stats to suptitle
    if stats is not None:
        stat_sum = ""
        types = ["bias", "corr", "ioa", "mse", "ss", "rmse"]
        if "dist" in stats:
            types += ["dist"]
        for type in types:
            stat_sum += f"{type}: {stats[type]['value']:.1f}  "

        title = f"{title}: {stat_sum}"

    fig.suptitle(title)  # , fontsize=fs_title, loc="left")

    # plt.tight_layout()
    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
