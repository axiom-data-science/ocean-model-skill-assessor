"""
Utility functions.
"""

import json
import logging
import pathlib
import sys

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cf_pandas as cfp
import cf_xarray
import extract_model as em
import intake
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from cf_pandas import Vocab, always_iterable, astype, merge
from intake.catalog import Catalog
from shapely.geometry import Polygon
from xarray import DataArray, Dataset

from .paths import Paths


def read_model_file(
    fname_processed_model: Path, no_Z: bool, dsm: xr.Dataset
) -> xr.Dataset:
    """_summary_

    Parameters
    ----------
    fname_processed_model : Path
        Model file path
    no_Z : bool
        _description_
    dsm : Dataset

    Returns
    -------
    Processed model output (Dataset)
    """

    model = xr.open_dataset(fname_processed_model).cf.guess_coord_axis()
    try:
        check_dataset(model, no_Z=no_Z)
    except KeyError:
        # see if I can fix it
        model = fix_dataset(model, dsm)
        check_dataset(model, no_Z=no_Z)

    return model


def read_processed_data_file(
    fname_processed_data: Path, no_Z: bool
) -> Union[xr.Dataset, pd.DataFrame]:
    """_summary_

    Parameters
    ----------
    fname_processed_data : Path
        Data file path
    no_Z : bool
        _description_

    Returns
    -------
    Processed data (DataFrame or Dataset)
    """

    # read in from newly made file to make sure output is loaded
    if ".csv" in str(fname_processed_data):
        obs = pd.read_csv(fname_processed_data)
        obs = check_dataframe(obs, no_Z)
    elif ".nc" in str(fname_processed_data):
        obs = xr.open_dataset(fname_processed_data).cf.guess_coord_axis()
        check_dataset(obs, is_model=False, no_Z=no_Z)
    else:
        raise TypeError("object is neither DataFrame nor Dataset.")

    return obs


def save_processed_files(
    dfd: Union[xr.Dataset, pd.DataFrame],
    fname_processed_data: Path,
    model_var: xr.Dataset,
    fname_processed_model: Path,
):
    """Save processed data and model output into files.

    Parameters
    ----------
    dfd : Union[xr.Dataset, pd.DataFrame]
        Processed data
    fname_processed_data : Path
        Data file path
    model_var : xr.Dataset
        Processed model output
    fname_processed_model : Path
        Model file path
    """

    if isinstance(dfd, pd.DataFrame):
        # # make sure datetimes will be recognized when reread
        # # actually seems to work without this
        # dfd = dfd.rename(columns={dfd.cf["T"].name: "time"})
        dfd.to_csv(fname_processed_data, index=False)
    elif isinstance(dfd, xr.Dataset):
        dfd.to_netcdf(fname_processed_data)
        dfd.close()
    else:
        raise TypeError("object is neither DataFrame nor Dataset.")
    if fname_processed_model.is_file():
        pathlib.Path.unlink(fname_processed_model)
    model_var.to_netcdf(fname_processed_model, mode="w")
    model_var.close()


def fix_dataset(
    model_var: Union[xr.DataArray, xr.Dataset], ds: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """Fill in info necessary to pass `check_dataset()` if possible.

    Right now it is only for converting horizontal indices to lon/lat but conceivably could do more in the future. Looks for lon/lat being 2D coords.

    Parameters
    ----------
    model_var : Union[xr.DataArray,xr.Dataset]
        xarray object that needs some more info filled in
    ds : Union[xr.DataArray,xr.Dataset]
        xarray object that has info that can be used to fill in model_var

    Returns
    -------
    Union[xr.DataArray,xr.Dataset]
        model_var with more information included, hopefully
    """

    # see if lon/lat are in model_var as data_vars instead of as coordinates
    if (
        "longitude" not in model_var.cf.coordinates and "longitude" in model_var.cf
    ) or ("latitude" not in model_var.cf.coordinates and "latitude" in model_var.cf):
        lonkey, latkey = model_var.cf["longitude"].name, model_var.cf["latitude"].name
        model_var = model_var.assign_coords(
            {lonkey: model_var[lonkey], latkey: model_var[latkey]}
        )

    # if we have X/Y indices in model_var but not their equivalent lon/lat, get them from ds
    elif (
        "longitude" not in model_var.cf.coordinates
        and "X" in model_var.cf
        and "longitude" in ds.cf.coordinates
        # and ds.cf["longitude"].ndim == 2
        and ds[cf_xarray.accessor._get_all(ds, "longitude")[0]].ndim == 2
        and "latitude" not in model_var.cf.coordinates
        and "Y" in model_var.cf
        and "latitude" in ds.cf
        # and ds.cf["latitude"].ndim == 2
        and ds[cf_xarray.accessor._get_all(ds, "latitude")[0]].ndim == 2
    ):
        lonkey, latkey = ds.cf["longitude"].name, ds.cf["latitude"].name
        X, Y = model_var.cf["X"], model_var.cf["Y"]
        # model_var[lonkey] = ds.cf["longitude"].isel({Y.name: Y, X.name: X})
        # model_var[lonkey].attrs = ds[lonkey].attrs
        model_var = model_var.assign_coords(
            {
                lonkey: ds.cf["longitude"].isel({Y.name: Y, X.name: X}),
                latkey: ds.cf["latitude"].isel({Y.name: Y, X.name: X}),
            }
        )

    # see if Z is in variables but not in coords
    # can't figure out how to catch this case but generalize yet
    if "Z" not in model_var.cf.coordinates and "s_rho" in model_var.variables:
        model_var = model_var.assign_coords({"s_rho": model_var["s_rho"]})

    return model_var


def check_dataset(
    ds: Union[xr.DataArray, xr.Dataset], is_model: bool = True, no_Z: bool = False
):
    """Check xarray datasets (usually model output) for necessary cf-xarray dims/coords.

    If Dataset is model output (`is_model=True`), must have T, Z, vertical, latitude, longitude, and "positive" attribute must be associated with Z or vertical. But, if `no_Z=True`, neither Z, vertical, nor positive attribute need to be present.

    If Dataset is not model output (is_model=False), must have T, Z, latitude, longitude. But, if `no_Z=True`, Z does not need to be present.

    """

    if "T" not in ds.cf:
        raise KeyError(
            "a variable of datetimes needs to be identifiable by `cf-xarray` in dataset. Ways to address this include: variable name has the word 'time' in it; variable contains datetime objects; variable has an attribute of `'axis': 'T'`. See `cf-xarray` docs for more information."
        )
    if not no_Z:
        if is_model:
            if "Z" not in ds.cf or "vertical" not in ds.cf:
                raise KeyError(
                    "a variable of depths needs to be identifiable by `cf-xarray` in dataset for both axis 'Z' and coordinate 'vertical'. Ways to address this include: variable name has the word 'depth' in it; for axis 'Z' variable has an attribute of `'axis': 'Z'`. See `cf-xarray` docs for more information."
                )
            if (
                "positive" not in ds[ds.cf.axes["Z"][0]].attrs
                and "positive" not in ds[ds.cf.coordinates["vertical"][0]].attrs
            ):
                raise KeyError(
                    "ds.cf['Z'] or ds.cf['vertical'] needs to have an attribute stating `'positive': 'up'` or `'positive': 'down'`."
                )
        else:
            if "Z" not in ds.cf:
                raise KeyError(
                    "a variable of depths needs to be identifiable by `cf-xarray` in dataset for axis 'Z'. Ways to address this include: variable name has the word 'depth' in it; variable has an attribute of `'axis': 'Z'`. See `cf-xarray` docs for more information."
                )

    if "longitude" not in ds.cf.coordinates or "latitude" not in ds.cf.coordinates:
        raise KeyError(
            "A variable containing longitudes and a variable containing latitudes must each be identifiable. One way to address this is to make sure the variable names start with 'lon' and 'lat' respectively. See `cf-xarray` docs for more information."
        )


def check_dataframe(dfd: pd.DataFrame, no_Z: bool) -> pd.DataFrame:
    """Check dataframe for T, Z, lon, lat; reset indices; parse dates."""

    # drop index if it is just the default range index, otherwise return to columns
    if (
        isinstance(dfd.index, pd.core.indexes.range.RangeIndex)
        and dfd.index.start == 0
        and dfd.index.stop == len(dfd.index)
    ):
        drop = True
    else:
        drop = False

    dfd = dfd.reset_index(drop=drop)

    # check for presence of required axis/coord information
    # in the future relax these requirements depending on featuretype is instead is in
    # catalog metadata
    if "T" not in dfd.cf:
        raise KeyError(
            "a column of datetimes needs to be identifiable by `cf-pandas` in dataset. One way to address this is to make sure the name of the column has the word 'time' in it."
        )
    if "Z" not in dfd.cf and not no_Z:
        raise KeyError(
            "a column of depths (even if the same value) needs to be identifiable by `cf-pandas` in dataset. If there is no concept of depth for this dataset, you can instead set `no_Z=True`. If a depth-column is present, make sure it has 'depth' in the column name."
        )
    if "longitude" not in dfd.cf or "latitude" not in dfd.cf:
        raise KeyError(
            "A column containing longitudes and a column containing latitudes must each be identifiable by `cf-pandas`. If they are present make sure the column name includes 'lon' and 'lat', respectively."
        )

    dfd[dfd.cf["T"].name] = pd.to_datetime(dfd.cf["T"])

    return dfd


def check_catalog(
    cat: Catalog,
    source_names: Optional[list] = None,
    skip_strings: Optional[list] = None,
):
    """Check a catalog for required keys.

    Parameters
    ----------
    catalogs : Catalog
        Catalog object
    source_names : list
        Use these source_names instead of list(cat) if entered, for checking.
    skip_strings : list of strings, optional
        If provided, source_names in catalog will only be checked for goodness if they do not contain one of skip_strings. For example, if `skip_strings=["_base"]` then any source in the catalog whose name contains that string will be skipped.

    """

    required_keys = {
        "minLongitude",
        "maxLongitude",
        "minLatitude",
        "maxLatitude",
        "minTime",
        "maxTime",
        "featuretype",
        "maptype",
    }

    skip_strings = skip_strings or []

    if source_names is None:
        source_names = list(cat)

    for skip_string in skip_strings:
        source_names = [
            source_name
            for source_name in source_names
            if skip_string not in source_name
        ]

    for source_name in source_names:
        missing_keys = set(required_keys) - set(cat[source_name].metadata.keys())

        if len(missing_keys) > 0:
            raise KeyError(
                f"In catalog {cat.name} and dataset {source_name}, missing required keys {missing_keys}."
            )

    allowed_featuretypes = [
        "timeSeries",
        "profile",
        "trajectoryProfile",
        "timeSeriesProfile",
        "grid",
    ]
    future_featuretypes = ["trajectory"]

    if cat[source_name].metadata["featuretype"] in future_featuretypes:
        raise KeyError(
            f"featuretype {cat[source_name].metadata['featuretype']} is not available yet."
        )
    elif cat[source_name].metadata["featuretype"] not in allowed_featuretypes:
        raise KeyError(
            f"featuretype in metadata must be one of {allowed_featuretypes} but instead is {cat[source_name].metadata['featuretype']}."
        )

    allowed_maptypes = ["point", "line", "box"]
    if cat[source_name].metadata["maptype"] not in allowed_maptypes:
        raise KeyError(
            f"maptype in metadata must be one of {allowed_maptypes} but instead is {cat[source_name].metadata['maptype']}."
        )


def open_catalogs(
    catalogs: Union[str, Catalog, Sequence],
    paths: Optional[Paths] = None,
    skip_check: bool = False,
    skip_strings: Optional[list] = None,
) -> List[Catalog]:
    """Initialize catalog objects from inputs.

    Parameters
    ----------
    catalogs : Union[str, Catalog, Sequence]
        Catalog name(s) or list of names, or catalog object or list of catalog objects.
    paths : Paths, optional
        Paths object for finding paths to use. Required if any catalog is a string referencing paths.
    skip_check : bool
        If True, do not check catalogs. Use this for testing as needed. Default is False.
    skip_strings : list of strings, optional
        If provided, source_names in catalog will only be checked for goodness if they do not contain one of skip_strings. For example, if `skip_strings=["_base"]` then any source in the catalog whose name contains that string will be skipped.

    Returns
    -------
    list[Catalog]
        Catalogs, ready to use.
    """

    catalogs = always_iterable(catalogs)
    cats = []
    for catalog in catalogs:
        if isinstance(catalog, str):
            if paths is None:
                raise KeyError("if any catalog is a string, need to input `paths`.")
            cat = intake.open_catalog(paths.CAT_PATH(catalog))
        elif isinstance(catalog, Catalog):
            cat = catalog
        else:
            raise ValueError(
                "Catalog(s) should be input as string paths or Catalog objects or Sequence thereof."
            )

        if not skip_check:
            check_catalog(cat, skip_strings=skip_strings)
        cats.append(cat)

    return cats


def open_vocabs(
    vocabs: Union[str, Vocab, Sequence, Path], paths: Optional[Paths] = None
) -> Vocab:
    """Open vocabularies, can input mix of forms.

    Parameters
    ----------
    vocabs : Union[str, Vocab, Sequence, Path]
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
    paths : Paths, optional
        Paths object for finding paths to use. Required if any input vocab is a str referencing paths.

    Returns
    -------
    Vocab
        Single Vocab object with vocab stored in vocab.vocab
    """
    vocab_objects = []
    vocabs = always_iterable(vocabs)
    for vocab in vocabs:
        # convert to Vocab object
        if isinstance(vocab, str):
            if paths is None:
                raise KeyError("if any vocab is a string, need to input `paths`.")
            vocab = Vocab(paths.VOCAB_PATH(vocab))
        elif isinstance(vocab, Path):
            vocab = Vocab(vocab)
        elif isinstance(vocab, Vocab):
            vocab = vocab
        else:
            raise ValueError(
                "Vocab(s) should be input as string, paths or Vocab objects or Sequence thereof."
            )
        vocab_objects.append(vocab)
    vocab = merge(vocab_objects)

    return vocab


def open_vocab_labels(
    vocab_labels: Union[str, dict, Path],
    paths: Optional[Paths] = None,
) -> dict:
    """Open dict of vocab_labels if needed

    Parameters
    ----------
    vocab_labels : Union[str, Vocab, Sequence, Path], optional
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
    paths : Paths, optional
        Paths object for finding paths to use.

    Returns
    -------
    dict
        dict of vocab_labels for plotting
    """

    if isinstance(vocab_labels, str):
        assert (
            paths is not None
        ), "need to input `paths` to `open_vocab_labels()` if inputting string."
        vocab_labels = json.loads(
            open(
                pathlib.Path(paths.VOCAB_PATH(vocab_labels)).with_suffix(".json"),
                "r",
            ).read()
        )
    elif isinstance(vocab_labels, Path):
        vocab_labels = json.loads(open(vocab_labels.with_suffix(".json"), "r").read())
    elif isinstance(vocab_labels, dict):
        vocab_labels = vocab_labels
    else:
        raise ValueError("vocab_labels should be input as string, Path, or dict.")
    assert isinstance(vocab_labels, dict)
    return vocab_labels


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
        logging.info("Lon/lat coordinates are 1D and are being changed to 2D.")

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


def set_up_logging(verbose, paths: Paths, mode: str = "w", testing: bool = False):
    """set up logging"""

    if not testing:
        logging.captureWarnings(True)

    file_handler = logging.FileHandler(filename=paths.LOG_PATH, mode=mode)
    handlers: List[Union[logging.StreamHandler, logging.FileHandler]] = [file_handler]
    if verbose:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        handlers.append(stdout_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d}\n%(levelname)s - %(message)s\n",
        handlers=handlers,
    )

    # logger = logging.getLogger('OMSA log')
    logger = logging.getLogger(__name__)

    return logger


def get_mask(
    dsm: Dataset, varname: str, wetdry: bool = False
) -> Union[DataArray, None]:
    """Return mask that matches x/y coords of var.

    If no mask can be identified with `.filter_by_attrs(flag_meanings="land water")`, instead will make one of non-nans for 1 horizontal grid cross-section of varname.

    Parameters
    ----------
    dsm : Dataset
        Model output
    varname : str
        Name of variable in dsm.
    wetdry : bool
        If True, selected mask must include "wetdry" in name and will use first time step.

    Returns
    -------
    DataArray
        mask associated with varname in dsm
    """

    logging.info("Retrieving mask")

    # if not varname in dsm.data_vars:
    #     raise KeyError(
    #         f"varname {varname} needs to be a data variable in dsm but is not found."
    #     )

    # include matching static mask if present
    masks = dsm.filter_by_attrs(flag_meanings="land water")
    # dask or something messes up the flags? just in case:
    if len(masks.data_vars) == 0:
        masks = dsm.filter_by_attrs(option_0="land")

    if wetdry:
        mask_names = [mask for mask in masks if "wetdry" in mask]
        masks = dsm[mask_names]

    if (len(masks.data_vars) > 0) and (varname in dsm.data_vars):
        if wetdry:
            mask_name = [
                mask
                for mask in masks.data_vars
                if dsm[mask].encoding["coordinates"]
                in dsm[varname].encoding["coordinates"]
                or dsm[mask].cf.isel(T=0).shape == dsm[varname].shape
            ][0]
            mask = dsm[mask_name].cf.isel(T=0)
        else:
            mask_name = [
                mask
                for mask in masks.data_vars
                if dsm[mask].encoding["coordinates"]
                in dsm[varname].encoding["coordinates"]
                or dsm[mask].shape == dsm[varname].shape
            ][0]
            mask = dsm[mask_name]
    elif (len(masks.data_vars) > 0) and (varname in dsm.coords):
        if wetdry:
            mask_name = [
                mask
                for mask in masks.data_vars
                if dsm[mask].cf.isel(T=0).shape == dsm[varname].shape
            ][0]
            mask = dsm[mask_name].cf.isel(T=0)
        else:
            mask_name = [
                mask
                for mask in masks.data_vars
                if dsm[mask].shape == dsm[varname].shape
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

    # if dask-backed, read into memory
    if mask.chunks is not None:
        mask = mask.load()
    return mask


def find_bbox(
    ds: xr.DataArray,
    paths: Optional[Paths] = None,
    mask: Optional[DataArray] = None,
    dd: int = 1,
    alpha: int = 5,
    save: bool = False,
) -> tuple:
    """Determine bounds and boundary of model.

    This does not know how to handle a rectilinear 1D lon/lat model with a mask

    Parameters
    ----------
    ds: DataArray
        xarray Dataset containing model output.
    paths : Paths
        Paths object for finding paths to use.
    mask : DataArray, optional
        Mask with 1's for active locations and 0's for masked.
    dd: int, optional
        Number to decimate model output lon/lat, as a stride.
    alpha: int, optional
        Number for alphashape to determine what counts as the convex hull. Larger number is more detailed, 1 is a good starting point.
    save : bool, optional
        Input True to save.

    Returns
    -------
    List
        Contains the name of the longitude and latitude variables for ds, geographic bounding box of model output (`[min_lon, min_lat, max_lon, max_lat]`), low res and high res wkt representation of model boundary.

    Notes
    -----
    This was originally from the package ``model_catalogs``.
    """

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

    # This is structured, rectilinear
    # GFS, RTOFS, HYCOM
    if (lon.ndim == 1) and ("nele" not in ds.dims):
        nlon, nlat = ds[lonkey].size, ds[latkey].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
        # boundary = np.vstack((lonb, latb)).T
        p = Polygon(zip(lonb, latb))
        p0 = p.simplify(1)
        # Now using the more simplified version because all of these models are boxes
        p1 = p0

        if mask is not None:
            raise NotImplemented

    else:

        if mask is not None:

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

        else:
            lon = lon.flatten()
            lat = lat.flatten()

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

    if save:
        if paths is None:
            words = "To save the model boundary, you need to input `paths`."
            raise ValueError(words)
        with open(paths.ALPHA_PATH, "w") as text_file:
            text_file.write(p1.wkt)

    # useful things to look at: p.wkt  #shapely.geometry.mapping(p)
    return lonkey, latkey, list(p0.bounds), p1


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
        lkey, xkey = dam.cf["longitude"].name, dam.cf["X"].name
        nlon = int((dam[lkey] >= 180).sum())  # number of longitudes to roll by
        dam = dam.assign_coords({lkey: (((dam[lkey] + 180) % 360) - 180)})
        # rotate arrays so that the locations and values are -180 to 180
        # instead of 0 to 180 to -180 to 0
        dam = dam.roll({xkey: nlon}, roll_coords=True)
        logging.warning(
            "Longitudes are being shifted because they look like they are not -180 to 180."
        )
    return dam


def kwargs_search_from_model(
    kwargs_search: Dict[str, Union[str, float]], paths: Paths
) -> dict:
    """Adds spatial and/or temporal range from model output to dict.

    Examines model output and uses the bounding box of the model as the search spatial range if needed, and the time range of the model as the search time search if needed. They are added into `kwargs_search` and the dict is returned.

    Parameters
    ----------
    kwargs_search : dict
        Keyword arguments to input to search on the server before making the catalog.
    paths : Paths
        Paths object for finding paths to use.

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
            model_cat = intake.open_catalog(paths.CAT_PATH(kwargs_search["model_name"]))
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


def calculate_anomaly(
    dd_in: Union[pd.Series, pd.DataFrame, xr.DataArray],
    monthly_mean,
    varname=None,
) -> pd.Series:
    """Given monthly mean that is indexed by month of year, subtract it from time series to get anomaly.

    Should work with both pd.Series/pd.DataFrame and xr. DataArray.
    Assume that variable in monthly_mean is the same as in the input time series.
    The way it works for DataArrays is by changing it to a DataFrame. Assumes this is a time series.

    Returns dd as the type as DataFrame it is came in as Series and Dataset if it came in DataArray. It is pd.Series in the middle so this probably won't work well for datasets that are more complex than time series.
    """

    if varname is None:
        varname = dd_in.name
    else:
        varname = dd_in.cf[
            varname
        ].name  # translate from key_variable alias to actual variable name
    varname_mean = f"{varname}_mean"
    varname_anomaly = f"{varname}_anomaly"

    # if monthly_mean is None:
    #     monthly_mean = dd[varname].groupby(dd.cf["T"].dt.month).mean()

    # in_type = type(dd)

    # if isinstance(dd, xr.DataArray):
    #     dd = dd.squeeze().to_dataframe()

    if isinstance(dd_in, pd.Series):
        dd_in = dd_in.to_frame()  # this changes dd into a DataFrame

    # import pdb; pdb.set_trace()

    dd = pd.DataFrame()
    dd["time"] = dd_in.cf["T"].values
    # dd["time"] = dd_in.index.values  # save times
    dd = dd.set_index(dd["time"].dt.month)
    dd[varname_mean] = monthly_mean
    dd = dd.set_index(dd["time"].name)

    # this shifts the mean for the first and last month so they are a bit off since they aren't interpolated
    # using the month before and month after, but the middle months are good.
    # generally this sets the mean to the 15th of the month rather than the beginning or end
    inan = (dd.index.day != 15) * (dd.index > dd.index[0]) * (dd.index < dd.index[-1])
    dd.loc[inan, varname_mean] = pd.NA

    inan = dd[varname_mean] == dd[varname_mean].shift(1)
    dd.loc[inan, varname_mean] = pd.NA

    dd[varname_mean] = dd[varname_mean].interpolate()
    dd[varname_anomaly] = dd_in[varname].squeeze() - dd[varname_mean]

    # return in original container
    if isinstance(dd_in, (xr.DataArray, xr.Dataset)):
        dd_out = xr.DataArray(
            coords={dd_in.cf["T"].name: dd.index.values},
            data=dd[varname_anomaly].values,
        ).broadcast_like(dd_in[varname])
        if len(dd_in[varname].coords) > len(dd_out.coords):
            coordstoadd = list(set(dd_in[varname].coords) - set(dd_out.coords))
            for coord in coordstoadd:
                dd_out[coord] = dd_in[varname][coord]
        dd_out.attrs = dd_in[varname].attrs
        dd_out.name = dd_in[varname].name

    elif isinstance(dd_in, (pd.Series, pd.DataFrame)):

        dd_out = pd.DataFrame()
        for key in ["T", "Z", "latitude", "longitude"]:
            dd_out[dd_in.cf[key].name] = dd_in.cf[key]
        dd_out[varname_anomaly] = dd[varname_anomaly]

    return dd_out


def calculate_distance(lons, lats):
    """Calculate distance (km), esp for transects."""

    G = pyproj.Geod(ellps="WGS84")
    distance = G.inv(
        lons[:-1],
        lats[:-1],
        lons[1:],
        lats[1:],
    )[2]
    distance = np.hstack((np.array([0]), distance))
    distance = distance.cumsum() / 1000  # km
    return distance
