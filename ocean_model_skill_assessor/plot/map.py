"""
Plot map.
"""

from pathlib import PurePath
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intake.catalog import Catalog
from matplotlib.pyplot import figure
from numpy import allclose, array, asarray
from shapely.geometry import Polygon
from xarray import DataArray, Dataset

from ..utils import astype, find_bbox, open_catalogs, shift_longitudes


try:
    import cartopy.crs

    CARTOPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CARTOPY_AVAILABLE = False  # pragma: no cover

col_label = "k"  # "r"
res = "10m"


def setup_ax(ax, land_10m, left_labels=True, fontsize=12):
    """Basic plot setup for map."""
    gl = ax.gridlines(
        linewidth=0.2, color="gray", alpha=0.5, linestyle="-", draw_labels=True
    )
    gl.bottom_labels = False  # turn off labels where you don't want them
    gl.right_labels = False
    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}
    if not left_labels:
        gl.left_labels = False
        gl.right_labels = True
    ax.coastlines(resolution=res)
    ax.add_feature(land_10m, facecolor="0.8")


def plot_map(
    maps: array,
    figname: Union[str, PurePath],
    extent: Optional[list] = None,
    p: Optional[Polygon] = None,
    label_with_station_name: bool = False,
    dd: list = [0.0, 0.0],
    annotate: bool = True,
    annotate_fontsize: int = 12,
    figsize: Tuple[int, int] = (8, 7),
    two_maps: dict = None,
    map_font_size: int = 12,
    markersize: int = 5,
    markeredgewidth: float = 0.5,
    linewidth_data: int = 3,
    linewidth_poly: int = 2,
    alpha_marker: float = 1.0,
    colors_data: Union[str, list] = col_label,
    legend: bool = False,
    loc: str = "best",
    suptitle: Optional[str] = None,
    suptitle_fontsize: int = 16,
    tight_layout: bool = True,
):
    """Plot and save to file map of model domain and data locations.

    Parameters
    ----------
    maps : array
        Info about datasets. [min_lon, max_lon, min_lat, max_lat, source_name]. Can have optional 6th element in the list for what type of representation to use in the plot: "point", "box", or "line". If the type isn't input and maxlons don't equal minlon, assume box to plot instead of line.
    figname : Union[str, PurePath]
        Map will be saved here.
    extent: optional
        [min longitude, max longitude, min latitude, max latitude]
    p : Shapely polygon
        Polygon representing outer boundary of numerical model.
    label_with_station_name : bool
        If True, use station names to label instead of a counter from 0 to number of stations - 1.
    dd : list
        Distance [x,y] to push the annotation away from the min lon/lat of a station. To input per station, make list of lists that is the same size as the number of stations.
    annotate : bool
        True to annotate.
    annotate_fontsize : int
        Fontsize for annotations
    figsize : tuple
        Figure size for matplotlib
    two_maps : dict
        Plot two maps side by side: presumably a zoomed-out version on the left ("extent_left") with a box showing the area enclosed by the magnified map on the right ("extent_right"). The data locations are only plotted in the right-hand map. You can also input "width_ratios" to shift the width between the map and the magnified map.

        Example usage: ``two_maps = dict(extent_left=[1,4,1,4], extent_right=[2,3,2,3], width_ratios=[0.67, 1.33])``
    map_font_size : int
        Font size for grid labels.
    markersize : int
        Markersize for points. Default is 5.
    markeredgewidth : float
        Edge width for markers for points (black line). Default is 0.5.
    linewidth_data : int
        Line width for plotting data when it involves lines.
    linewidth_poly : int
        Line width for plotting polygon.
    alpha_marker: float
        alpha for markers for points
    colors_data : str
        One color to use for all or colors in a list matching number of stations.
    legend : bool
        True for legend instead of annotations.
    loc : str
        legend location for matplotlib. Default "best".
    suptitle : str
        Title for top of the figure, overall.
    suptitle_fontsize : int
        Fontsize for suptitle. Default is 16.
    tight_layout : bool
        Whether to use tight_layout when have 2 maps.
    """

    if not CARTOPY_AVAILABLE:
        raise ModuleNotFoundError(  # pragma: no cover
            "Cartopy is not available so map will not be plotted."
        )

    import cartopy

    pc = cartopy.crs.PlateCarree()
    land_10m = cartopy.feature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="face", facecolor="0.8"
    )

    min_lons, max_lons = maps[:, 0].astype(float), maps[:, 1].astype(float)
    min_lats, max_lats = maps[:, 2].astype(float), maps[:, 3].astype(float)
    station_names = maps[:, 4].astype(str)

    # if isinstance(colors_data, str):
    #     mat = np.ones((len(min_lons)), dtype=str)
    #     mat[:] = colors_data
    #     colors_data = mat
    #     # colors_data = [colors_data]*len(min_lons)
    # else:
    #     colors_data = colors_data

    # check for presence of type
    if maps.shape[1] > 5:
        types = maps[:, 5].astype(str)

    central_longitude = min_lons.mean()
    proj = cartopy.crs.Mercator(central_longitude=float(central_longitude))

    fig = figure(figsize=figsize, dpi=100)
    if two_maps is not None:
        col_box = "#DE3163"
        # set up larger map
        if "width_ratios" in two_maps:
            width_ratios = two_maps["width_ratios"]
        else:
            width_ratios = [1, 1]
        ax_map, ax = fig.subplots(
            1,
            2,
            width_ratios=width_ratios,
            subplot_kw=dict(projection=proj, frameon=False),
        )
        setup_ax(ax_map, land_10m, fontsize=map_font_size)
        ax_map.set_extent(two_maps["extent_left"], pc)

        ax_map.set_frame_on(True)

        # add box
        import matplotlib.patches as mpatches

        extent = two_maps["extent_right"]
        assert isinstance(extent, list)
        ax_map.add_patch(
            mpatches.Rectangle(
                xy=[extent[0], extent[2]],
                width=extent[1] - extent[0],
                height=extent[3] - extent[2],
                facecolor="none",
                # alpha=0.5,
                lw=2,
                edgecolor=col_box,
                transform=pc,
                zorder=10,
            ),
        )

        # set up magnified map, which will be used for the rest of the function
        ax = fig.add_subplot(1, 2, 2, projection=proj)
        setup_ax(ax, land_10m, left_labels=False, fontsize=map_font_size)
        # add box to magnified plot to emphasize connection
        ax.add_patch(
            mpatches.Rectangle(
                xy=[extent[0], extent[2]],
                width=extent[1] - extent[0],
                height=extent[3] - extent[2],
                facecolor="none",
                # alpha=0.5,
                lw=4,
                edgecolor=col_box,
                transform=pc,
                zorder=10,
            ),
        )

    else:
        ax = fig.add_axes([0.06, 0.01, 0.93, 0.95], projection=proj)
        setup_ax(ax, land_10m)

    # alphashape
    if p is not None:
        # this needs to be checked for resulting type and order
        bbox = p.bounds
        ax.add_geometries(
            [p],
            crs=pc,
            facecolor="none",
            edgecolor="r",
            linestyle="-",
            linewidth=linewidth_poly,
        )
    else:
        bbox = [min(min_lons), min(min_lats), max(max_lons), max(max_lats)]

    kwargs_plot: Dict[str, Union[str, list]] = {}

    # plot stations
    inds = (min_lons == max_lons) | (types == "point")
    if inds.sum() > 0:
        if legend:
            kwargs_plot["label"] = list(station_names[inds])
            assert isinstance(colors_data, list)
            ax.set_prop_cycle(color=colors_data)
        else:
            kwargs_plot["color"] = colors_data
        # # import pdb; pdb.set_trace()
        # ax.scatter(
        #      min_lons[inds][:,np.newaxis],
        #     # min_lons[inds],
        #      min_lats[inds][:,np.newaxis],
        #     # min_lats[inds],
        #     # marker="o",
        #     s=markersize,
        #     c = colors_data,
        #     transform=pc,
        #     ls="",
        #     **kwargs_plot,
        # )
        ax.plot(
            [min_lons[inds]],
            #  min_lons[inds][:,np.newaxis],
            # min_lons[inds],
            [min_lats[inds]],
            #  min_lats[inds][:,np.newaxis],
            # min_lats[inds],
            marker="o",
            markersize=markersize,
            markeredgecolor="k",
            markeredgewidth=markeredgewidth,
            # color = colors_data,
            transform=pc,
            ls="",
            alpha=alpha_marker,
            **kwargs_plot,
        )

    # plot lines
    inds = types == "line"
    if inds.sum() > 0:
        if legend:
            kwargs_plot["label"] = list(station_names[inds])
            ax.set_prop_cycle(color=colors_data)
        else:
            kwargs_plot["color"] = colors_data  # [inds]
        ax.plot(
            [min_lons[inds], max_lons[inds]],
            [min_lats[inds], max_lats[inds]],
            linestyle="-",
            linewidth=linewidth_data,
            transform=pc,
            **kwargs_plot,
        )

    # plot boxes
    inds = types == "box"
    if inds.sum() > 0:
        if legend:
            kwargs_plot["label"] = list(station_names[inds])
            ax.set_prop_cycle(color=colors_data)
        else:
            kwargs_plot["color"] = colors_data  # [inds]
        ax.plot(
            [
                min_lons[inds],
                max_lons[inds],
                max_lons[inds],
                min_lons[inds],
                min_lons[inds],
            ],
            [
                min_lats[inds],
                min_lats[inds],
                max_lats[inds],
                max_lats[inds],
                min_lats[inds],
            ],
            linestyle="-",
            linewidth=linewidth_data,
            transform=pc,
            # label=station_names[inds],
            **kwargs_plot,
        )

    # annotate stations
    if not legend or annotate:
        if not isinstance(dd, list):
            raise ValueError("dd must be a list of distance in [x,y].")
        if isinstance(dd[0], (int, float)):
            dds = [dd] * len(min_lons)
        else:
            dds = dd
        for i, (lon, lat, dd) in enumerate(zip(min_lons, min_lats, dds)):
            if label_with_station_name:
                label = station_names[i]
            else:
                label = i
            xyproj = np.array(ax.projection.transform_point(lon, lat, pc)) + dd
            ax.annotate(
                label,
                xy=xyproj,
                xytext=xyproj,
                color=col_label,
                fontsize=annotate_fontsize,
            )

    # [min lon, max lon, min lat, max lat]
    if extent is None:
        extent_use = [bbox[0] - 0.1, bbox[2] + 0.1, bbox[1] - 0.1, bbox[3] + 0.1]
    else:
        extent_use = extent

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

    if legend:
        ax.legend(loc=loc, fontsize=annotate_fontsize)
        ax.set_prop_cycle(color=colors_data)

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize)

    if two_maps is not None and tight_layout:
        fig.tight_layout()

    fig.savefig(figname, dpi=100, bbox_inches="tight")


def plot_cat_on_map(
    catalog: Union[Catalog, str],
    project_name: str,
    figname: Optional[str] = None,
    remove_duplicates=None,
    **kwargs_map,
):
    """Plot catalog on map with optional model domain polygon.

    Parameters
    ----------
    catalog : Union[Catalog,str]
        Which catalog of datasets to plot on map.
    project_name : str
        name of project in case we need to find the project files.
    remove_duplicates : bool
        If True, take the set of the source in catalog based on the spatial locations so they are not repeated in the map.
    remove_duplicates : function, optional
        Input a function that takes in maps and return maps, and in between removes duplicate entries.

    Examples
    --------

    After creating catalog with `intake-erddap`, look at data locations:

    >>> omsa.plot.map.plot_cat_on_map(catalog=catalog_name, project_name=project_name)
    """

    cat = open_catalogs(catalog, project_name)[0]

    figname = figname or f"map_of_{cat.name}"

    # kwargs_map: Dict

    maps = []
    maps.extend(
        [
            [
                cat[s].metadata["minLongitude"],
                cat[s].metadata["maxLongitude"],
                cat[s].metadata["minLatitude"],
                cat[s].metadata["maxLatitude"],
                s,
                cat[s].metadata["maptype"] or "",
            ]
            for s in list(cat)
            if "minLongitude" in cat[s].metadata
        ]
    )
    # import pdb; pdb.set_trace()

    maps = asarray(maps)

    if remove_duplicates is not None:
        maps = remove_duplicates(maps)

    # if remove_duplicates:
    #     mapsdf = pd.DataFrame(maps)
    #     # create column on which to base dropping duplicates (by transect name)
    #     mapsdf["chooser"] = mapsdf[4].str.split("-").str.get(0)
    #     maps = mapsdf.drop_duplicates(subset=["chooser"]).to_numpy()

    plot_map(maps, figname, **kwargs_map)
