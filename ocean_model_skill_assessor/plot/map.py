"""
Plot map.
"""

import cartopy
import matplotlib.pyplot as plt
import numpy as np


pc = cartopy.crs.PlateCarree()
col_label = "r"
land_10m = cartopy.feature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="face", facecolor="0.8"
)


def plot(
    lls,
    names,
    boundary,
    proj=None,
    res="110m",
    extent=None,
    figname="figure.png",
    dpi=100,
):
    """Plot data locations on map.

    Parameters
    ----------
    lls: array-like, Nx2
        Data locations. lls[:,0] longitudes and lls[:,1] latitudes.
    names: str, list-like
        Names of data locations to use as labels on map.
    boundary: array-like, Nx2
        Model boundary locations. boundary[:,0] longitudes and boundary[:,1]
        latitudes.
    proj: proj instance
        Projection from cartopy. Example: `cartopy.crs.Mercator()`.
    res: str
        Resolution of Natural Earth features. Options: '110m', '50m', '10m'.
    extent: list
        Extent of domain as [min lon, max lon, min lat, max lat].
    figname: str
        Filename for figure (as absolute or relative path).
    dpi: int
        dpi for figure.
    """

    # enforce longs/lats/names all same length
    # enforce type
    lons, lats = np.asarray(lls[:, 0]), np.asarray(lls[:, 1])

    if not isinstance(names, list):
        names = [names]

    if not proj:
        central_longitude = lons.mean()
        proj = cartopy.crs.Mercator(central_longitude=central_longitude)

    fig = plt.figure(figsize=(8, 7), dpi=100)
    ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=proj)
    # ax.set_frame_on(False) # kind of like it without the box
    # ax.set_extent([-98, -87.5, 22.8, 30.5], cartopy.crs.PlateCarree())
    gl = ax.gridlines(
        linewidth=0.2, color="gray", alpha=0.5, linestyle="-", draw_labels=True
    )
    gl.bottom_labels = False  # turn off labels where you don't want them
    gl.right_labels = False
    ax.coastlines(resolution=res)
    ax.add_feature(land_10m, facecolor="0.8")

    ax.plot(*boundary.T, color="0.7", ls=":", transform=pc)

    ax.plot(lons, lats, marker="o", transform=pc, ls="", color=col_label)

    for name, lon, lat in zip(names, lons, lats):
        xyproj = ax.projection.transform_point(lon, lat, pc)
        xyprojshift = ax.projection.transform_point(lon + 0.1, lat + 0.1, pc)
        ax.annotate(name, xy=xyproj, xytext=xyprojshift, color=col_label)

    if extent:
        ax.set_extent(extent, pc)

    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
