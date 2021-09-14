"""
Plot map.
"""

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

pc = cartopy.crs.PlateCarree()
col_label = "r"
land_10m = cartopy.feature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="face", facecolor="0.8"
)


def plot(
    lls_stations=None,
    names_stations=None,
    lls_boxes=None,
    names_boxes=None,
    boundary=None,
    proj=None,
    res="110m",
    extent=None,
    figname=None,
    dpi=100,
):
    """Plot data locations on map.

    Parameters
    ----------
    lls_stations: array-like, Nx2
        Data locations. lls_stations[:,0] longitudes and lls_stations[:,1] latitudes.
    names_stations: str, list-like
        Names of station data locations to use as labels on map.
    lls_boxes
    names_boxes
    boundary: array-like, Nx2, optional
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
    
    if figname is None:
        figname = "figure.png"

    # enforce longs/lats/names all same length
    # enforce type
#     import pdb; pdb.set_trace()
    if boundary is not None:
        central_longitude = boundary[:,0].mean()
    elif lls_stations is not None:
        central_longitude = lls_stations[:, 0].mean()
    elif lls_boxes is not None:
        central_longitude = lls_boxes[:, 0].mean()

    if not isinstance(names_stations, list):
        names_stations = [names_stations]

    if not proj:
#         central_longitude = lons.mean()
        proj = cartopy.crs.Mercator(central_longitude=central_longitude)
#     import pdb; pdb.set_trace()
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

    if boundary is not None:
        ax.plot(*boundary.T, color="0.7", ls=":", transform=pc)

    # plot stations
    if lls_stations is not None:
        lons, lats = np.asarray(lls_stations[:, 0]), np.asarray(lls_stations[:, 1])
        ax.plot(lons, lats, marker="o", transform=pc, ls="", color=col_label)
    
    # Plot boxes
    if lls_boxes is not None:
        for box, name in zip(lls_boxes, names_boxes):
            ax.add_patch(mpatches.Rectangle(xy=box[:2], 
                                             width=box[2]-box[0], 
                                             height=box[3]-box[1],
                                             linewidth=1,
                                             edgecolor='r',
                                             facecolor='none',
                                             transform=pc,
                                             zorder=10
                                            )
                         )
            xyproj = ax.projection.transform_point(*box[:2], pc)
            ax.annotate(name, xy=xyproj, xytext=xyproj, color=col_label)
    
    if lls_stations is not None:
        for name, lon, lat in zip(names_stations, lons, lats):
            xyproj = ax.projection.transform_point(lon, lat, pc)
            xyprojshift = ax.projection.transform_point(lon + 0.1, lat + 0.1, pc)
            ax.annotate(name, xy=xyproj, xytext=xyprojshift, color=col_label)

    if extent:
        ax.set_extent(extent, pc)

    fig.savefig(figname, dpi=dpi, bbox_inches="tight")
