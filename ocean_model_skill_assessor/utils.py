"""
Utility functions.
"""

import logging
import sys

from pathlib import PurePath
from typing import Dict, List, Optional, Sequence, Union

import cf_pandas as cfp
import extract_model as em
import intake
import numpy as np
import xarray as xr

from cf_pandas import Vocab, always_iterable, astype, merge
from intake.catalog import Catalog
from shapely.geometry import Polygon
from xarray import DataArray, Dataset

from .paths import CAT_PATH, LOG_PATH, VOCAB_PATH


def open_catalogs(
    catalogs: Union[str, Catalog, Sequence], project_name: str
) -> List[Catalog]:
    """Initialize catalog objects from inputs.

    Parameters
    ----------
    catalogs : Union[str, Catalog, Sequence]
        Catalog name(s) or list of names, or catalog object or list of catalog objects.
    project_name : str
        Subdirectory in cache dir to store files associated together.

    Returns
    -------
    list[Catalog]
        Catalogs, ready to use.
    """

    catalogs = always_iterable(catalogs)
    if isinstance(catalogs[0], str):
        cats = [
            intake.open_catalog(CAT_PATH(catalog_name, project_name))
            for catalog_name in astype(catalogs, list)
        ]
    elif isinstance(catalogs[0], Catalog):
        cats = catalogs
    else:
        raise ValueError(
            "Catalog(s) should be input as string paths or Catalog objects or Sequence thereof."
        )

    return cats


def open_vocabs(vocabs: Union[str, Vocab, Sequence, PurePath]) -> Vocab:
    """_summary_

    Parameters
    ----------
    vocabs : Union[str, Vocab, Sequence, PurePath]
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.

    Returns
    -------
    Vocab
        Single Vocab object with vocab stored in vocab.vocab
    """

    vocabs = always_iterable(vocabs)
    if isinstance(vocabs[0], str):
        vocab = merge([Vocab(VOCAB_PATH(v)) for v in vocabs])
    elif isinstance(vocabs[0], PurePath):
        vocab = merge([Vocab(v) for v in vocabs])
    elif isinstance(vocabs[0], Vocab):
        vocab = merge(vocabs)
    else:
        raise ValueError(
            "Vocab(s) should be input as string paths or Vocab objects or Sequence thereof."
        )

    return vocab


def coords1Dto2D(dam: DataArray) -> DataArray:
    """expand 1D coordinates to 2D

    Parameters
    ----------
    dam : DataArray
        Model output variable to work on.

    Returns
    -------
    DataArray
        Model output but with 2D coordinates in place of 1D coordinates, if applicable. Otherwise same as input.
    """

    if dam.cf["longitude"].ndim == 1:
        # need to meshgrid lon/lat
        lon2, lat2 = np.meshgrid(dam.cf["longitude"], dam.cf["latitude"])
        lonkey, latkey = dam.cf["longitude"].name, dam.cf["latitude"].name
        # 2D coord names
        lonkey2, latkey2 = f"{lonkey}2", f"{latkey}2"
        # dam = dam.assign_coords({lonkey2: ((dam.cf["Y"].name, dam.cf["X"].name), lon2, dam.cf["Longitude"].attrs),
        #                          latkey2: ((dam.cf["Y"].name, dam.cf["X"].name), lat2, dam.cf["Latitude"].attrs)})
        dam[lonkey2] = (
            (dam.cf["Y"].name, dam.cf["X"].name),
            lon2,
            dam.cf["Longitude"].attrs,
        )
        dam[latkey2] = (
            (dam.cf["Y"].name, dam.cf["X"].name),
            lat2,
            dam.cf["Latitude"].attrs,
        )

        # remove attributes from 1D lon/lats that are interpreted for coordinates (but not for axes)
        if "standard_name" in dam[lonkey].attrs:
            del dam[lonkey].attrs["standard_name"]
        if "units" in dam[lonkey].attrs:
            del dam[lonkey].attrs["units"]
        if "standard_name" in dam[latkey].attrs:
            del dam[latkey].attrs["standard_name"]
        if "units" in dam[latkey].attrs:
            del dam[latkey].attrs["units"]

        # modify coordinates
        if "_CoordinateAxes" in dam.attrs:
            dam.attrs["_CoordinateAxes"] = dam.attrs["_CoordinateAxes"].replace(
                lonkey, lonkey2
            )
            dam.attrs["_CoordinateAxes"] = dam.attrs["_CoordinateAxes"].replace(
                latkey, latkey2
            )
        elif "coordinates" in dam.encoding:
            dam.encoding["coordinates"] = dam.encoding["coordinates"].replace(
                lonkey, lonkey2
            )
            dam.encoding["coordinates"] = dam.encoding["coordinates"].replace(
                latkey, latkey2
            )

    return dam


def set_up_logging(project_name, verbose, mode: str = "w", testing: bool = False):
    """set up logging"""

    if not testing:
        logging.captureWarnings(True)

    file_handler = logging.FileHandler(filename=LOG_PATH(project_name), mode=mode)
    handlers: List[Union[logging.StreamHandler, logging.FileHandler]] = [file_handler]
    if verbose:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        handlers.append(stdout_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # logger = logging.getLogger('OMSA log')


def get_mask(dsm: Dataset, varname: str) -> Union[DataArray, None]:
    """Return mask that matches x/y coords of var.

    If no mask can be identified with `.filter_by_attrs(flag_meanings="land water")`, instead will make one of non-nans for 1 horizontal grid cross-section of varname.

    Parameters
    ----------
    dsm : Dataset
        Model output
    varname : str
        Name of variable in dsm.

    Returns
    -------
    DataArray
        mask associated with varname in dsm
    """

    if not varname in dsm.data_vars:
        raise KeyError(
            f"varname {varname} needs to be a data variable in dsm but is not found."
        )

    # include matching static mask if present
    masks = dsm.filter_by_attrs(flag_meanings="land water")
    if len(masks.data_vars) > 0:
        mask_name = [
            mask
            for mask in masks.data_vars
            if dsm[mask].encoding["coordinates"] in dsm[varname].encoding["coordinates"]
        ][0]
        mask = dsm[mask_name]
    else:
        # want just X and Y to make mask. Just use first time and surface depth value, as needed.
        dam = dsm[varname]
        if "T" in dam.cf.axes:
            dam = dam.cf.isel(T=0)
        if "Z" in dam.cf.axes:
            dam = dam.cf.sel(Z=0, method="nearest")

        mask = dam.notnull().load().astype(int)
        msg = "Generated mask for model using 1 horizontal cross section of model output and searching for nans."
        logging.info(msg)

    return mask


def shift_longitudes(dam: Union[DataArray, Dataset]) -> Union[DataArray, Dataset]:
    """Shift longitudes from 0 to 360 to -180 to 180 if necessary.

    Parameters
    ----------
    dam : Union[DataArray,Dataset]
        Object with model output to check

    Returns
    -------
    Union[DataArray,Dataset]
        Return model output with shifted longitudes, if it was necessary.
    """

    if dam.cf["longitude"].max() > 180:
        lkey = dam.cf["longitude"].name
        nlon = int((dam[lkey] >= 180).sum())  # number of longitudes to roll by
        dam = dam.assign_coords(lon=(((dam[lkey] + 180) % 360) - 180))
        # rotate arrays so that the locations and values are -180 to 180
        # instead of 0 to 180 to -180 to 0
        dam = dam.roll(lon=nlon, roll_coords=True)
        logging.warning(
            "Longitudes are being shifted because they look like they are not -180 to 180."
        )
    return dam


def find_bbox(
    ds: xr.DataArray, mask: Optional[DataArray] = None, dd: int = 1, alpha: int = 5
) -> tuple:
    """Determine bounds and boundary of model.

    Parameters
    ----------
    ds: DataArray
        xarray Dataset containing model output.
    mask : DataArray, optional
        Mask with 1's for active locations and 0's for masked.
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
    This was originally from the package ``model_catalogs``.
    """

    if mask is not None:
        hasmask = True
    else:
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
            lonkey = ds.cf.coordinates["longitude"][0]
            # need to make sure latkey matches lonkey grid
            latkey = f"lat{lonkey[3:]}"
        # In case there are multiple grids, just take first one;
        # they are close enough
        lon = ds[lonkey].values
        lat = ds[latkey].values

    if hasmask:

        if mask.ndim == 2 and lon.ndim == 1:
            # # need to meshgrid lon/lat
            # lon, lat = np.meshgrid(lon, lat)
            # This shouldn't happen anymore, so make note if it does
            msg = "1D coordinates were found for this model but that should not be possible anymore."
            raise ValueError(msg)

        lon = lon[np.where(mask == 1)]
        lon = lon[~np.isnan(lon)].flatten()
        lat = lat[np.where(mask == 1)]
        lat = lat[~np.isnan(lat)].flatten()

    # This is structured, rectilinear
    # GFS, RTOFS, HYCOM
    if (lon.ndim == 1) and ("nele" not in ds.dims) and not hasmask:
        nlon, nlat = ds[lonkey].size, ds[latkey].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
        # boundary = np.vstack((lonb, latb)).T
        p = Polygon(zip(lonb, latb))
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
        `kwargs_search` but with modifications if relevant.

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
        if isinstance(kwargs_search["model_name"], str):
            model_cat = intake.open_catalog(
                CAT_PATH(kwargs_search["model_name"], kwargs_search["project_name"])
            )
        elif isinstance(kwargs_search["model_name"], Catalog):
            model_cat = kwargs_search["model_name"]
        else:
            raise ValueError(
                "model_name should be input as string path or Catalog object."
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
