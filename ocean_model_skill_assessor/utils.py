"""
Utility functions.
"""

from typing import Dict, Optional, Union

import cf_pandas as cfp
import cf_xarray
import extract_model as em
import intake
import numpy as np
import shapely.geometry
import xarray as xr

import ocean_model_skill_assessor as omsa


def find_bbox(ds: xr.DataArray, dd: int = 1, alpha: int = 5) -> tuple:
    """Determine bounds and boundary of model.

    Parameters
    ----------
    ds: DataArray
        xarray Dataset containing model output.
    dd: int, optional
        Number to decimate model output lon/lat, as a stride.
    alpha: float, optional
        Number for alphashape to determine what counts as the convex hull. Larger number is more detailed, 1 is a good starting point.

    Returns
    -------
    List
        Contains the name of the longitude and latitude variables for ds, geographic bounding box of model output (`[min_lon, min_lat, max_lon, max_lat]`), low res and high res wkt representation of model boundary.

    Notes
    -----
    This is from the package model_catalogs.
    """

    hasmask = False

    try:
        lon = ds.cf["longitude"].values
        lat = ds.cf["latitude"].values
        lonkey = ds.cf["longitude"].name
        latkey = ds.cf["latitude"].name

    except KeyError:
        if "lon_rho" in ds:
            lonkey = "lon_rho"
            latkey = "lat_rho"
        else:
            lonkey = list(ds.cf[["longitude"]].coords.keys())[0]
            # need to make sure latkey matches lonkey grid
            latkey = f"lat{lonkey[3:]}"
        # In case there are multiple grids, just take first one;
        # they are close enough
        lon = ds[lonkey].values
        lat = ds[latkey].values

    # this function is being used on DataArrays instead of Datasets, and the model I'm using as
    # an example doesn't have a mask, so bring this back when I have a relevant example.
    # check for corresponding mask (rectilinear and curvilinear grids)
    if any([var for var in ds.data_vars if "mask" in var]):
        if ("mask_rho" in ds) and (lonkey == "lon_rho"):
            maskkey = lonkey.replace("lon", "mask")
        elif "mask" in ds:
            maskkey = "mask"
        else:
            maskkey = None
        if maskkey in ds:
            lon = ds[lonkey].where(ds[maskkey] == 1).values
            lon = lon[~np.isnan(lon)].flatten()
            lat = ds[latkey].where(ds[maskkey] == 1).values
            lat = lat[~np.isnan(lat)].flatten()
            hasmask = True
    # import pdb; pdb.set_trace()
    # This is structured, rectilinear
    # GFS, RTOFS, HYCOM
    if (lon.ndim == 1) and ("nele" not in ds.dims) and not hasmask:
        nlon, nlat = ds["lon"].size, ds["lat"].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
        # boundary = np.vstack((lonb, latb)).T
        p = shapely.geometry.Polygon(zip(lonb, latb))
        p0 = p.simplify(1)
        # Now using the more simplified version because all of these models are boxes
        p1 = p0

    elif hasmask or ("nele" in ds.dims):  # unstructured

        assertion = (
            "dd and alpha need to be defined in the catalog metadata for this model."
        )
        assert dd is not None and alpha is not None, assertion

        # this leads to a circular import error if read in at top level bc of other packages brought in.
        import alphashape

        lon, lat = lon[::dd], lat[::dd]
        pts = list(zip(lon, lat))

        # need to calculate concave hull or alphashape of grid
        # low res, same as convex hull
        p0 = alphashape.alphashape(pts, 0.0)
        # downsample a bit to save time, still should clearly see shape of domain
        # import pdb; pdb.set_trace()
        # pts = shapely.geometry.MultiPoint(list(zip(lon, lat)))
        p1 = alphashape.alphashape(pts, alpha)

    # else:  # 2D coordinates

    #     # this leads to a circular import error if read in at top level bc of other packages brought in.
    #     import alphashape

    #     lon, lat = lon.flatten()[::dd], lat.flatten()[::dd]

    #     # need to calculate concave hull or alphashape of grid
    #     # low res, same as convex hull
    #     p0 = alphashape.alphashape(list(zip(lon, lat)), 0.0)
    #     # downsample a bit to save time, still should clearly see shape of domain
    #     pts = shapely.geometry.MultiPoint(list(zip(lon, lat)))
    #     p1 = alphashape.alphashape(pts, alpha)

    # useful things to look at: p.wkt  #shapely.geometry.mapping(p)
    return lonkey, latkey, list(p0.bounds), p1


def kwargs_search_from_model(kwargs_search: Dict[str, Union[str, float]]) -> dict:
    """Adds spatial and/or temporal range from model output to dict.

    Examines model output and uses the bounding box of the model as the search spatial range if needed, and the time range of the model as the search time search if needed. They are added into `kwargs_search` and the dict is returned.

    Parameters
    ----------
    kwargs_search : dict
        Keyword arguments to input to search on the server before making the catalog.

    Returns
    -------
    dict
        kwargs_search but with modifications if relevant.

    Raises
    ------
    KeyError
        If all of `max_lon`, `min_lon`, `max_lat`, `min_lat` and `min_time`, `max_time` are already specified along with `model_name`.
    """

    # if model_name input, use it to select the search kwargs
    if "model_name" in kwargs_search:
        if kwargs_search.keys() >= {
            "max_lon",
            "min_lon",
            "min_lat",
            "max_lat",
            "min_time",
            "max_time",
        }:
            raise KeyError(
                "Can input `model_name` to `kwargs_search` to determine the spatial and/or temporal search box OR specify `max_lon`, `min_lon`, `max_lat`, `min_lat` and `min_time`, `max_time`. Can also do a combination of the two."
            )

        # read in model output
        model_cat = intake.open_catalog(
            omsa.CAT_PATH(kwargs_search["model_name"], kwargs_search["project_name"])
        )
        dsm = model_cat[list(model_cat)[0]].to_dask()

        kwargs_search.pop("model_name")
        kwargs_search.pop("project_name")

        # if none of these present, read from model output
        if kwargs_search.keys().isdisjoint(
            {
                "max_lon",
                "min_lon",
                "min_lat",
                "max_lat",
            }
        ):
            min_lon, max_lon = float(
                dsm[dsm.cf.coordinates["longitude"][0]].min()
            ), float(dsm[dsm.cf.coordinates["longitude"][0]].max())
            min_lat, max_lat = float(
                dsm[dsm.cf.coordinates["latitude"][0]].min()
            ), float(dsm[dsm.cf.coordinates["latitude"][0]].max())

            if abs(min_lon) > 180 or abs(max_lon) > 180:
                min_lon -= 360
                max_lon -= 360

            kwargs_search.update(
                {
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                }
            )

        if kwargs_search.keys().isdisjoint(
            {
                "max_time",
                "min_time",
            }
        ):
            min_time, max_time = str(dsm.cf["T"].min().values), str(
                dsm.cf["T"].max().values
            )

            kwargs_search.update(
                {
                    "min_time": min_time,
                    "max_time": max_time,
                }
            )

    return kwargs_search
