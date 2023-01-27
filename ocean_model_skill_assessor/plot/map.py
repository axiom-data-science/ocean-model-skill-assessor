"""
Plot map.
"""

from pathlib import PurePath
from typing import Dict, Optional, Sequence, Union

from intake.catalog import Catalog
from matplotlib.pyplot import figure
from numpy import allclose, array, asarray
from shapely.geometry import Polygon
from xarray import DataArray, Dataset

from ..utils import find_bbox, open_catalogs, shift_longitudes


try:
    import cartopy.crs

    CARTOPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CARTOPY_AVAILABLE = False  # pragma: no cover


def plot_map(
    maps: array,
    figname: Union[str, PurePath],
    extent: Optional[Sequence] = None,
    p: Optional[Polygon] = None,
):
    """Plot and save to file map of model domain and data locations.

    Parameters
    ----------
    maps : array
        Info about datasets. [min_lon, max_lon, min_lat, max_lat, source_name]
    figname : Union[str, PurePath]
        Map will be saved here.
    extent: optional
        [min longitude, max longitude, min latitude, max latitude]
    p : Shapely polygon
        Polygon representing outer boundary of numerical model.
    """

    if not CARTOPY_AVAILABLE:
        raise ModuleNotFoundError(  # pragma: no cover
            "Cartopy is not available so map will not be plotted."
        )

    import cartopy

    pc = cartopy.crs.PlateCarree()
    col_label = "k"  # "r"
    land_10m = cartopy.feature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="face", facecolor="0.8"
    )
    res = "10m"

    min_lons, max_lons = maps[:, 0].astype(float), maps[:, 1].astype(float)
    min_lats, max_lats = maps[:, 2].astype(float), maps[:, 3].astype(float)

    central_longitude = min_lons.mean()
    proj = cartopy.crs.Mercator(central_longitude=float(central_longitude))
    fig = figure(figsize=(8, 7), dpi=100)
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

    # alphashape
    if p is not None:
        # this needs to be checked for resulting type and order
        bbox = p.bounds
        ax.add_geometries([p], crs=pc, facecolor="none", edgecolor="r", linestyle="-")
    else:
        bbox = [min(min_lons), min(min_lats), max(max_lons), max(max_lats)]

    # plot stations
    # if min_lons == max_lons:  #  check these are stations
    ax.plot(
        min_lons,
        min_lats,
        marker="o",
        markersize=1,
        transform=pc,
        ls="",
        color=col_label,
    )

    # annotate stations
    for i, (lon, lat) in enumerate(zip(min_lons, min_lats)):
        xyproj = ax.projection.transform_point(lon, lat, pc)
        ax.annotate(i, xy=xyproj, xytext=xyproj, color=col_label)

    # [min lon, max lon, min lat, max lat]
    if extent is None:
        extent_use = [bbox[0] - 0.1, bbox[2] + 0.1, bbox[1] - 0.1, bbox[3] + 0.1]

    # if model is global - based on extent - write that it is global and use smaller extent
    if (
        allclose(bbox[0], -180, atol=2)
        and allclose(bbox[2], 180, atol=2)
        and allclose(bbox[1], -90, atol=2)
        and allclose(bbox[3], 90, atol=2)
    ):
        # explain global model
        ax.set_title("Only showing part of global model")

        # change delta deg for extent to max(10% of total diff lons/lats, 1 deg)
        if extent is None:
            dlon, dlat = 0.1 * (min(min_lons) - max(max_lons)), 0.1 * (
                min(min_lats) - max(max_lats)
            )
            ddlon, ddlat = max(dlon, 5), max(dlat, 2)
            extent_use = [
                min(min_lons) - ddlon,
                max(max_lons) + ddlon,
                min(min_lats) - ddlat,
                max(max_lats) + ddlat,
            ]

    ax.set_extent(extent_use, pc)

    fig.savefig(figname, dpi=100, bbox_inches="tight")


def plot_cat_on_map(
    catalog: Union[Catalog, str],
    project_name: str,
    figname: Optional[str] = None,
    **kwargs,
):
    """Plot catalog on map with optional model domain polygon.

    Parameters
    ----------
    catalog : Union[Catalog,str]
        Which catalog of datasets to plot on map.
    project_name : str
        name of project in case we need to find the project files.

    Examples
    --------

    After creating catalog with `intake-erddap`, look at data locations:

    >>> omsa.plot.map.plot_cat_on_map(catalog=catalog_name, project_name=project_name)
    """

    cat = open_catalogs(catalog, project_name)[0]

    figname = figname or f"map_of_{cat.name}"

    kwargs_map: Dict = {}

    maps = []
    maps.extend(
        [
            [
                cat[s].metadata["minLongitude"],
                cat[s].metadata["maxLongitude"],
                cat[s].metadata["minLatitude"],
                cat[s].metadata["maxLatitude"],
                s,
            ]
            for s in list(cat)
        ]
    )

    plot_map(asarray(maps), figname, **kwargs_map)
