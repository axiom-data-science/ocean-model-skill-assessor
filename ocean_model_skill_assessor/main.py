"""
Main run functions.
"""

import logging
import mimetypes
import pathlib
import warnings

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any, Dict, List, Optional, Union

import cf_xarray
import extract_model as em
import extract_model.accessor
import intake
import numpy as np
import pandas as pd
import requests
import shapely.wkt
import xarray as xr
import yaml

from cf_pandas import Vocab, astype
from cf_pandas import set_options as cfp_set_options
from cf_xarray import set_options as cfx_set_options
from datetimerange import DateTimeRange
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
from pandas import DataFrame, to_datetime
from shapely.geometry import Point

from ocean_model_skill_assessor.plot import map

from .paths import (
    ALIGNED_CACHE_DIR,
    ALPHA_PATH,
    CAT_PATH,
    MASK_PATH,
    MODEL_CACHE_DIR,
    OUT_DIR,
    PROJ_DIR,
    VOCAB_PATH,
)
from .stats import _align, save_stats
from .utils import (
    coords1Dto2D,
    find_bbox,
    get_mask,
    kwargs_search_from_model,
    open_catalogs,
    open_vocabs,
    set_up_logging,
    shift_longitudes,
)


# turn off annoying warning in cf-xarray
cfx_set_options(warn_on_missing_variables=False)


def make_local_catalog(
    filenames: List[str],
    filetype: Optional[str] = None,
    name: str = "local_catalog",
    description: str = "Catalog of user files.",
    metadata: dict = None,
    metadata_catalog: dict = None,
    skip_entry_metadata: bool = False,
    kwargs_open: Optional[Dict] = None,
) -> Catalog:
    """Make an intake catalog from specified data files, including model output locations.

    Pass keywords for xarray for model output into the catalog through ``kwargs_xarray``.

    ``kwargs_open`` and ``metadata`` must be the same for all filenames. If it is not, make multiple catalogs and you can input them individually into the run command.

    Parameters
    ----------
    filenames : list of paths
        Where to find dataset(s) from which to make local catalog.
    filetype : str, optional
        Type of the input filenames, if you don't want the function to try to guess. Must be in the form that can go into intake as f"open_{filetype}".
    name : str, optional
        Name for catalog.
    description : str, optional
        Description for catalog.
    metadata : dict, optional
        Metadata for individual source. If input dataset does not include the longitude and latitude position(s), you will need to include it in the metadata as keys `minLongitude`, `minLatitude`, `maxLongitude`, `maxLatitude`.
    metadata_catalog : dict, optional
        Metadata for catalog.
    kwargs_open : dict, optional
        Keyword arguments to pass on to the appropriate ``intake`` ``open_*`` call for model or dataset.
    skip_entry_metadata : bool, optional
        This is useful for testing in which case we don't want to actually read the file. If you are making a catalog file for a model, you may want to set this to `True` to avoid reading it all in for metadata.

    Returns
    -------
    Catalog
        Intake catalog with an entry for each dataset represented by a filename.

    Examples
    --------

    Make catalog to represent local or remote files with specific locations:

    >>> make_local_catalog([filename1, filename2])

    Make catalog to represent model output:

    >>> make_local_catalog([model output location], skip_entry_metadata=True, kwargs_open={"drop_variables": "tau"})
    """

    metadata = metadata or {}
    metadata_catalog = metadata_catalog or {}

    kwargs_open = kwargs_open or {}

    # if any of kwargs_open came in with "None" instead of None because of CLI, change back to None
    kwargs_open.update({key: None for key, val in kwargs_open.items() if val == "None"})

    sources = []
    for filename in filenames:
        mtype = mimetypes.guess_type(filename)[0]
        if filetype is not None:
            source = getattr(intake, f"open_{filetype}")(
                filename, **kwargs_open
            )  # , csv_kwargs=kwargs_open)
        elif (
            (mtype is not None and ("csv" in mtype or "text" in mtype))
            or "csv" in filename
            or "text" in filename
        ):
            source = getattr(intake, "open_csv")(filename, csv_kwargs=kwargs_open)
        elif ("thredds" in filename and "dodsC" in filename) or "dods" in filename:
            # use netcdf4 engine if not input in kwargs_xarray
            kwargs_open.setdefault("engine", "netcdf4")
            source = getattr(intake, "open_opendap")(filename, **kwargs_open)
        elif (
            (mtype is not None and "netcdf" in mtype)
            or "netcdf" in filename
            or ".nc" in filename
        ):
            source = getattr(intake, "open_netcdf")(filename, **kwargs_open)

        # combine input metadata with source metadata
        source.metadata.update(metadata)

        sources.append(source)

    # create dictionary of catalog entries
    entries = {
        PurePath(source.urlpath).stem: LocalCatalogEntry(
            name=PurePath(source.urlpath).stem,
            description=source.description if source.description is not None else "",
            driver=source._yaml()["sources"][source.name]["driver"],
            args=source._yaml()["sources"][source.name]["args"],
            metadata=source.metadata,
            direct_access="allow",
        )
        for i, source in enumerate(sources)
    }

    # create catalog
    cat = Catalog.from_dict(
        entries,
        name=name,
        description=description,
        metadata=metadata_catalog,
    )

    # now that catalog is made, go through sources and add metadata
    for source in list(cat):
        if not skip_entry_metadata:
            dd = cat[source].read()

            # only read lon/lat from file if didn't input lon/lat info
            if cat[source].metadata.keys() >= {
                "maxLongitude",
                "minLongitude",
                "minLatitude",
                "maxLatitude",
            }:
                dd["longitude"] = cat[source].metadata["minLongitude"]
                dd["latitude"] = cat[source].metadata["minLatitude"]
                cat[source].metadata = {
                    "minLongitude": cat[source].metadata["minLongitude"],
                    "minLatitude": cat[source].metadata["minLatitude"],
                    "maxLongitude": cat[source].metadata["minLongitude"],
                    "maxLatitude": cat[source].metadata["minLatitude"],
                }

            else:
                metadata = {
                    "minLongitude": float(dd.cf["longitude"].min()),
                    "minLatitude": float(dd.cf["latitude"].min()),
                    "maxLongitude": float(dd.cf["longitude"].max()),
                    "maxLatitude": float(dd.cf["latitude"].max()),
                }

            # set up some basic metadata for each source
            dd.cf["T"] = to_datetime(dd.cf["T"])
            dd.set_index(dd.cf["T"], inplace=True)
            if dd.index.tz is not None:
                # logger is already defined in other function
                logger.warning(  # type: ignore
                    "Dataset %s had a timezone %s which is being removed. Make sure the timezone matches the model output.",
                    source,
                    str(dd.index.tz),
                )
                dd.index = dd.index.tz_convert(None)
                dd.cf["T"] = dd.index
            metadata.update(
                {
                    "minTime": str(dd.cf["T"].min()),
                    "maxTime": str(dd.cf["T"].max()),
                }
            )

            cat[source].metadata.update(metadata)
            cat[source]._entry._metadata.update(metadata)

    # create dictionary of catalog entries
    sources = [cat[source] for source in list(cat)]
    entries = {
        PurePath(source.urlpath).stem: LocalCatalogEntry(
            name=PurePath(source.urlpath).stem,
            description=source.description if source.description is not None else "",
            driver=source._yaml()["sources"][source.name]["driver"],
            args=source._yaml()["sources"][source.name]["args"],
            metadata=source.metadata,
            direct_access="allow",
        )
        for i, source in enumerate(sources)
    }

    # create catalog
    cat = Catalog.from_dict(
        entries,
        name=name,
        description=description,
        metadata=metadata_catalog,
    )

    return cat


def make_catalog(
    catalog_type: str,
    project_name: str,
    catalog_name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    kwargs_search: Optional[Dict[str, Union[str, int, float]]] = None,
    kwargs_open: Optional[Dict] = None,
    vocab: Optional[Union[Vocab, str, PurePath]] = None,
    return_cat: bool = True,
    save_cat: bool = False,
    verbose: bool = True,
    mode: str = "w",
    testing: bool = False,
):
    """Make a catalog given input selections.

    Parameters
    ----------
    catalog_type : str
        Which type of catalog to make? Options are "erddap", "axds", or "local".
    project_name : str
        Subdirectory in cache dir to store files associated together.
    catalog_name : str, optional
        Catalog name, with or without suffix of yaml. Otherwise a default name based on the catalog type will be used.
    description : str, optional
        Description for catalog.
    metadata : dict, optional
        Catalog metadata.
    kwargs : dict, optional
        Available keyword arguments for catalog types. Find more information about options in the original docs for each type. Some inputs might be required, depending on the catalog type.
    kwargs_search : dict, optional
        Keyword arguments to input to search on the server before making the catalog. These are not used with ``make_local_catalog()``; only for catalog types "erddap" and "axds".
        Options are:

        * to search by bounding box: include all of min_lon, max_lon, min_lat, max_lat: (int, float). Longitudes must be between -180 to +180.
        * to search within a datetime range: include both of min_time, max_time: interpretable datetime string, e.g., "2021-1-1"
        * to search using a textual keyword: include `search_for` as a string.
        * model_name can be input in place of either the spatial box or the time range or both in which case those values will be found from the model output. model_name should match a catalog file in the directory described by project_name.

    kwargs_open : dict, optional
        Keyword arguments to save into local catalog for model to pass on to ``xr.open_mfdataset`` call or ``pandas`` ``open_csv``. Only for use with ``catalog_type=local``.
    vocab : str, Vocab, Path, optional
        Way to find the criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for.
    return_cat : bool, optional
        Return catalog. For when using as a Python package instead of with command line.
    save_cat: bool, optional
        Save catalog to disk into project directory under `catalog_name`.
    verbose : bool, optional
        Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log.
    mode : str, optional
        mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.
    testing : boolean, optional
        Set to True if testing so warnings come through instead of being logged.
    """

    logger = set_up_logging(project_name, verbose, mode=mode, testing=testing)

    if kwargs_search is not None and catalog_type == "local":
        warnings.warn(
            "`kwargs_search` were input but will not be used since `catalog_type=='local'`.",
            UserWarning,
        )

    if kwargs_open is not None and catalog_type != "local":
        warnings.warn(
            f"`kwargs_open` were input but will not be used since `catalog_type=={catalog_type}`.",
            UserWarning,
        )

    kwargs = kwargs or {}
    kwargs_search = kwargs_search or {}

    # get spatial and/or temporal search terms from model if desired
    kwargs_search.update({"project_name": project_name})
    if catalog_type != "local":
        kwargs_search = kwargs_search_from_model(kwargs_search)

    if vocab is not None:
        if isinstance(vocab, str):
            vocab = Vocab(VOCAB_PATH(vocab))
        elif isinstance(vocab, PurePath):
            vocab = Vocab(vocab)
        elif isinstance(vocab, Vocab):
            pass
        else:
            raise ValueError("Vocab should be input as string, Path, or Vocab object.")

    if description is None:
        description = f"Catalog of type {catalog_type}."

    if catalog_type == "local":
        catalog_name = "local_cat" if catalog_name is None else catalog_name
        if "filenames" not in kwargs:
            raise ValueError("For `catalog_type=='local'`, must input `filenames`.")
        filenames = kwargs["filenames"]
        kwargs.pop("filenames")
        cat = make_local_catalog(
            astype(filenames, list),
            name=catalog_name,
            description=description,
            metadata=metadata,
            kwargs_open=kwargs_open,
            **kwargs,
        )

    elif catalog_type == "erddap":
        catalog_name = "erddap_cat" if catalog_name is None else catalog_name
        if "server" not in kwargs:
            raise ValueError("For `catalog_type=='erddap'`, must input `server`.")
        if vocab is not None:
            with cfp_set_options(custom_criteria=vocab.vocab):
                cat = intake.open_erddap_cat(
                    kwargs_search=kwargs_search,
                    name=catalog_name,
                    description=description,
                    metadata=metadata,
                    **kwargs,
                )
        else:
            cat = intake.open_erddap_cat(
                kwargs_search=kwargs_search,
                name=catalog_name,
                description=description,
                metadata=metadata,
                **kwargs,
            )

    elif catalog_type == "axds":
        catalog_name = "axds_cat" if catalog_name is None else catalog_name
        if vocab is not None:
            with cfp_set_options(custom_criteria=vocab.vocab):
                cat = intake.open_axds_cat(
                    kwargs_search=kwargs_search,
                    name=catalog_name,
                    description=description,
                    metadata=metadata,
                    **kwargs,
                )
        else:
            cat = intake.open_axds_cat(
                kwargs_search=kwargs_search,
                name=catalog_name,
                description=description,
                metadata=metadata,
                **kwargs,
            )

    if save_cat:
        # save cat to file
        cat.save(CAT_PATH(catalog_name, project_name))
        logger.info(
            f"Catalog saved to {CAT_PATH(catalog_name, project_name)} with {len(list(cat))} entries."
        )

    # logger.shutdown()

    if return_cat:
        return cat


def run(
    catalogs: Union[str, Catalog, Sequence],
    project_name: str,
    key_variable: Union[str, dict],
    model_name: Union[str, Catalog],
    vocabs: Union[str, Vocab, Sequence, PurePath],
    ndatasets: Optional[int] = None,
    kwargs_map: Optional[Dict] = None,
    verbose: bool = True,
    mode: str = "w",
    testing: bool = False,
    alpha: int = 5,
    dd: int = 2,
    preprocess: bool = False,
    need_xgcm_grid: bool = False,
    kwargs_xroms: Optional[dict] = None,
    interpolate_horizontal: bool = True,
    horizontal_interp_code="delaunay",
    want_vertical_interp: Optional[bool] = None,
    extrap: bool = False,
    model_source_name: Optional[str] = None,
    catalog_source_names=None,
    user_min_time: Optional[Union[str, pd.Timestamp]] = None,
    user_max_time: Optional[Union[str, pd.Timestamp]] = None,
    check_in_boundary: bool = True,
    tidal_filtering: Optional[Dict[str, bool]] = None,
    ts_mods: list = None,
    model_only: bool = False,
    plot_map: bool = True,
    no_Z: bool = False,
    wetdry: bool = False,
    plot_count_title: bool = True,
    **kwargs,
):
    """Run the model-data comparison.

    Note that timezones are assumed to match between the model output and data.

    Parameters
    ----------
    catalogs : str, list, Catalog
        Catalog name(s) or list of names, or catalog object or list of catalog objects. Datasets will be accessed from catalog entries.
    project_name : str
        Subdirectory in cache dir to store files associated together.
    key_variable : str, dict
        Key in vocab(s) representing variable to compare between model and datasets.
    model_name : str, Catalog
        Name of catalog for model output, created with ``make_catalog`` call, or Catalog instance.
    vocabs : str, list, Vocab, PurePath, optional
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
    ndatasets : int, optional
        Max number of datasets from each input catalog to use.
    kwargs_map : dict, optional
        Keyword arguments to pass on to ``omsa.plot.map.plot_map`` call.
    verbose : bool, optional
        Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log.
    mode : str, optional
        mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.
    testing : boolean, optional
        Set to True if testing so warnings come through instead of being logged.
    alpha : int
        parameter for alphashape. 0 returns qhull, and higher values make a tighter polygon around the points.
    dd : int
        number to decimate model points by when calculating model boundary with alphashape. input 1 to not decimate.
    preprocess : bool, optional
        If True, use function from ``extract_model`` to preprocess model output.
    need_xgcm_grid: bool
        If True, try to set up xgcm grid for run, which will be used for the variable calculation for the model.
    kwargs_xroms : dict
        Optional keyword arguments to pass to xroms.open_dataset
    interpolate_horizontal : bool, optional
        If True, interpolate horizontally. Otherwise find nearest model points.
    horizontal_interp_code: str
        Default "xesmf" to use package ``xESMF`` for horizontal interpolation, which is probably better if you need to interpolate to many points. To use ``xESMF`` you have install it as an optional dependency. Input "tree" to use BallTree to find nearest 3 neighbors and interpolate using barycentric coordinates. This has been tested for interpolating to 3 locations so far. Input "delaunay" to use a delaunay triangulation to find the nearest triangle points and interpolate the same as with "tree" using barycentric coordinates. This should be faster when you have more points to interpolate to, especially if you save and reuse the triangulation.
    want_vertical_interp: optional, bool
        This is None unless the user wants to specify that vertical interpolation should happen. This is used in only certain cases but in those cases it is important so that it is known to interpolate instead of try to figure out a vertical level index (which is not possible currently).
    extrap: bool
        Passed to `extract_model.select()`. Defaults to False. Pass True to extrapolate outside the model domain.
    model_source_name : str, optional
        Use this to access a specific source in the input model_catalog instead of otherwise just using the first source in the catalog.
    catalog_source_names

    user_min_time : str, optional
        If this is input, it will be used as the min time for the model
    user_max_time : str, optional
        If this is input, it will be used as the max time for the model
    check_in_boundary : bool
        If True, station location will be compared against model domain polygon to check if inside domain. Set to False to skip this check which might be desirable if you want to just compare with the closest model point.
    tidal_filtering: dict,
        ``tidal_filtering["model"]=True`` to tidally filter modeling output after em.select() is run, and ``tidal_filtering["data"]=True`` to tidally filter data.
    ts_mods

    model_only: bool
        If True, reads in model output and saves to cache, then stops. Default False.
    plot_map : bool
        If False, don't plot map
    no_Z : bool
        If True, set Z=None so no vertical interpolation or selection occurs. Do this if your variable has no concept of depth, like the sea surface height.
    wetdry : bool
        If True, insist that masked used has "wetdry" in the name and then use the first time step of that mask.
    plot_count_title : bool
        If True, have a count to match the map of the station number in the title, like "0: [station name]". Otherwise skip count.
    """

    logger = set_up_logging(project_name, verbose, mode=mode, testing=testing)

    logger.info(f"Input parameters: {locals()}")

    kwargs_map = kwargs_map or {}

    mask = None

    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    vocab = open_vocabs(vocabs)

    # Open catalogs.
    cats = open_catalogs(catalogs, project_name)

    # Warning about number of datasets
    ndata = np.sum([len(list(cat)) for cat in cats])
    if ndatasets is not None:
        logger.info(
            f"Note that we are using {ndatasets} datasets of {ndata} datasets. This might take awhile."
        )
    else:
        logger.info(
            f"Note that there are {ndata} datasets to use. This might take awhile."
        )

    # initialize model Dataset as None to compare with later
    # don't open model output at all if not needed (because it has already been saved, for example)
    dsm = None
    preprocessed = False
    p1 = None

    # loop over catalogs and sources to pull out lon/lat locations for plot
    maps = []
    count = 0  # track datasets since count is used to match on map
    for cat in cats:
        logger.info(f"Catalog {cat}.")
        if catalog_source_names is not None:
            source_names = catalog_source_names
        else:
            source_names = list(cat)
        for i, source_name in enumerate(source_names[:ndatasets]):

            if ndatasets is None:
                msg = (
                    f"\nsource name: {source_name} ({i+1} of {ndata} for catalog {cat}."
                )
            else:
                msg = f"\nsource name: {source_name} ({i+1} of {ndatasets} for catalog {cat}."
            logger.info(msg)

            if (
                "key_variables" in cat[source_name].metadata
                and key_variable not in cat[source_name].metadata["key_variables"]
            ):
                continue

            min_lon = cat[source_name].metadata["minLongitude"]
            max_lon = cat[source_name].metadata["maxLongitude"]
            min_lat = cat[source_name].metadata["minLatitude"]
            max_lat = cat[source_name].metadata["maxLatitude"]

            new_map = [min_lon, max_lon, min_lat, max_lat, source_name]
            # include maptype if available
            if "maptype" in cat[source_name].metadata:
                new_map += [cat[source_name].metadata["maptype"]]
            maps.append(new_map)

            # first loop dsm should be None
            # this is just a simple connection, no extra processing etc
            if dsm is None:
                # read in model output
                model_cat = open_catalogs(model_name, project_name)[0]
                if model_source_name is not None:
                    dsm = model_cat[model_source_name].to_dask()
                else:
                    dsm = model_cat[list(model_cat)[0]].to_dask()

                # the main preprocessing happens later, but do a minimal job here
                # so that cf-xarray can be used hopefully
                dsm = em.preprocess(dsm)

            # Do min and max separately.
            # min
            # if user_min_time is not None:
            #     user_min_time = user_min_time
            # else:
            model_min_time = pd.Timestamp(str(dsm.cf["T"][0].values))
            user_min_time = pd.Timestamp(user_min_time)

            if "minTime" in cat[source_name].metadata:
                data_min_time = cat[source_name].metadata["minTime"]
            # use kwargs_search min/max times if available
            elif (
                "kwargs_search" in cat.metadata
                and "min_time" in cat.metadata["kwargs_search"]
            ):
                data_min_time = cat.metadata["kwargs_search"]["min_time"]
            else:
                raise KeyError("Need a way to input min time desired.")

            # max
            # if user_max_time is not None:
            #     user_max_time = user_max_time
            # else:
            model_max_time = pd.Timestamp(str(dsm.cf["T"][-1].values))
            user_max_time = pd.Timestamp(user_max_time)

            if "maxTime" in cat[source_name].metadata:
                data_max_time = cat[source_name].metadata["maxTime"]
            # use kwargs_search min/max times if available
            elif (
                "kwargs_search" in cat.metadata
                and "max_time" in cat.metadata["kwargs_search"]
            ):
                data_max_time = cat.metadata["kwargs_search"]["max_time"]
            else:
                raise KeyError("Need a way to input max time desired.")
            # remove "Z" from min_time, max_time if present since assuming all in UTC
            data_min_time = pd.Timestamp(data_min_time.replace("Z", ""))
            data_max_time = pd.Timestamp(data_max_time.replace("Z", ""))

            # take time constraints as min/max if available and more constricting
            if (
                "constraints" in cat[source_name].describe()["args"]
                and "time>=" in cat[source_name].describe()["args"]["constraints"]
            ):
                constrained_min_time = pd.Timestamp(
                    cat[source_name]
                    .describe()["args"]["constraints"]["time>="]
                    .replace("Z", "")
                )
                if constrained_min_time > data_min_time:
                    data_min_time = constrained_min_time
            if (
                "constraints" in cat[source_name].describe()["args"]
                and "time<=" in cat[source_name].describe()["args"]["constraints"]
            ):
                constrained_max_time = pd.Timestamp(
                    cat[source_name]
                    .describe()["args"]["constraints"]["time<="]
                    .replace("Z", "")
                )
                if constrained_max_time < data_max_time:
                    data_max_time = constrained_max_time

            logger.info(
                f"""
                            User time range: {user_min_time} to {user_max_time}.
                            Model time range: {model_min_time} to {model_max_time}.
                            Data time range: {data_min_time} to {data_max_time}.
                            Data lon range: {min_lon} to {max_lon}.
                            Data lat range: {min_lat} to {max_lat}."""
            )

            # allow for possibility that key_variable is a dict with more complicated usage than just a string
            if isinstance(key_variable, dict):
                key_variable_data = key_variable["data"]
            else:
                key_variable_data = key_variable

            # Combine and align the two time series of variable
            with cfp_set_options(custom_criteria=vocab.vocab):

                # skip this dataset if times between data and model don't align

                data_time_range = DateTimeRange(data_min_time, data_max_time)
                model_time_range = DateTimeRange(model_min_time, model_max_time)
                user_time_range = DateTimeRange(user_min_time, user_max_time)
                if not data_time_range.is_intersection(model_time_range):
                    msg = f"Time range of dataset {source_name} and model output do not overlap. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue
                if (
                    pd.notnull(user_min_time)
                    and pd.notnull(user_max_time)
                    and not data_time_range.is_intersection(user_time_range)
                ):
                    msg = f"Time range of dataset {source_name} and user-input time range do not overlap. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue
                # in certain cases, the user input time range might be outside of the model availability
                if (
                    pd.notnull(user_min_time)
                    and pd.notnull(user_max_time)
                    and not model_time_range.is_intersection(user_time_range)
                ):
                    logger.warning(
                        "User-input time range is outside of model availability, so moving on..."
                    )
                    continue

                try:
                    dfd = cat[source_name].read()

                except requests.exceptions.HTTPError as e:
                    logger.warning(str(e))
                    msg = f"Data cannot be loaded for dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue

                # Need to have this here because if model file has previously been read in but
                # aligned file doesn't exist yet, this needs to run to update the sign of the
                # data depths in certain cases.
                # sort out depths between model and data
                # 1 location: interpolate or nearest neighbor horizontally
                # have it figure out depth
                if ("Z" not in dfd.cf.axes) or no_Z:
                    Z = None
                    vertical_interp = False
                    logger.info(
                        f"Will not perform vertical interpolation and there is no concept of depth for {key_variable_data}."
                    )
                elif (dfd.cf["Z"] == 0).all():
                    Z = 0  # do nearest depth to 0
                    vertical_interp = False
                    logger.info(
                        f"Will not perform vertical interpolation and will find nearest depth to {Z}."
                    )

                # if depth varies in time and will interpolate to match depths
                elif (dfd.cf["Z"] != dfd.cf["Z"][0]).any() and want_vertical_interp:
                    zkeym = dsm.cf.coordinates["vertical"][0]

                    # if the model depths are positive up/negative down, make sure the data match
                    if isinstance(dfd, (xr.DataArray, xr.Dataset)):
                        attrs = dfd[dfd.cf["Z"].name].attrs
                        if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                            encoding = dfd[dfd.cf["Z"].name].encoding

                        if dsm[zkeym].attrs["positive"] == "up":
                            dfd[dfd.cf["Z"].name] = np.negative(dfd.cf["Z"])
                        else:
                            dfd[dfd.cf["Z"].name] = np.positive(dfd.cf["Z"])

                        dfd.cf["Z"].attrs = attrs
                        if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                            dfd.cf["Z"].encoding = encoding

                    elif isinstance(dfd, (pd.DataFrame, pd.Series)):
                        if dsm[zkeym].attrs["positive"] == "up":
                            ilev = dfd.index.names.index(dfd.cf["Z"].name)
                            dfd.index = dfd.index.set_levels(
                                np.negative(abs(dfd.index.levels[ilev])), level=ilev
                            )
                            # depth might also be a column in addition to being in the index
                            if dfd.cf["Z"].name in dfd.columns:
                                dfd.cf["Z"] = np.negative(abs(dfd.cf["Z"]))
                            # dfd.cf["Z"] = np.negative(dfd.cf["Z"].values)
                            # dfd[zkey] = np.negative(dfd[zkey].values)
                        else:
                            ilev = dfd.index.names.index(dfd.cf["Z"].name)
                            dfd.index = dfd.index.set_levels(
                                np.positive(abs(dfd.index.levels[ilev])), level=ilev
                            )
                            # depth might also be a column in addition to being in the index
                            if dfd.cf["Z"].name in dfd.columns:
                                dfd.cf["Z"] = np.positive(abs(dfd.cf["Z"]))
                            # dfd.cf["Z"] = np.positive(dfd.cf["Z"].values)
                            # dfd[zkey] = np.positive(dfd[zkey].values)
                            # ilev = dfd.index.names.index(index.name)
                            # dfd.index = dfd.index.set_levels(index, level=ilev)

                    # if isinstance(dfd, (xr.DataArray, xr.Dataset)):
                    #     dfd.cf["Z"].attrs = attrs
                    #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                    #         dfd.cf["Z"].encoding = encoding
                    Z = dfd.cf["Z"].values
                    vertical_interp = True
                    logger.info(f"Will perform vertical interpolation, to depths {Z}.")

                # if depth varies in time and need to determine depth index
                else:
                    # elif (dfd.cf["Z"] != dfd.cf["Z"][0]).any():
                    # elif (dfd.cf["Z"] != dfd.cf["Z"].mean()).any():
                    # warnings.warn("Method to find index for depth not at surface not available yet.")
                    # raise UserWarning("Method to find index for depth not at surface not available yet.")
                    raise NotImplementedError(
                        "Method to find index for depth not at surface not available yet."
                    )

                    # if not need_xgcm_grid:
                    #     raise ValueError("Need xgcm, so input ``need_xgcm_grid==True``.")
                    # Z = dfd.cf["Z"].mean()
                    # vertical_interp = False
                    # zkeym = dsm.cf.coordinates["vertical"][0]
                    # if dsm[zkeym].attrs["positive"] == "up":
                    #     Z = np.negative(Z)
                    # else:
                    #     Z = np.positive(Z)
                    # # depths = dsm.z_rho0[:,ie,ix].squeeze().load()
                    # # iz = int(np.absolute(np.absolute(depths) - np.absolute(mean_depth)).argmin().values)
                # else:

                # zkey = dfd.cf["Z"].name
                # zkeym = dsm.cf.coordinates["vertical"][0]

                # # if the model depths are positive up/negative down, make sure the data match
                # if isinstance(dfd, (xr.DataArray, xr.Dataset)):
                #     attrs = dfd[dfd.cf["Z"].name].attrs
                #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                #         encoding = dfd[dfd.cf["Z"].name].encoding

                #     if dsm[zkeym].attrs["positive"] == "up":
                #         dfd[dfd.cf["Z"].name] = np.negative(dfd.cf["Z"])
                #     else:
                #         dfd[dfd.cf["Z"].name] = np.positive(dfd.cf["Z"])

                #     dfd.cf["Z"].attrs = attrs
                #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                #         dfd.cf["Z"].encoding = encoding

                # elif isinstance(dfd, (pd.DataFrame, pd.Series)):

                #     if dsm[zkeym].attrs["positive"] == "up":
                #         ilev = dfd.index.names.index(dfd.cf["Z"].name)
                #         dfd.index = dfd.index.set_levels(np.negative(dfd.index.levels[ilev]), level=ilev)
                #         # dfd.cf["Z"] = np.negative(dfd.cf["Z"].values)
                #         # dfd[zkey] = np.negative(dfd[zkey].values)
                #     else:
                #         ilev = dfd.index.names.index(dfd.cf["Z"].name)
                #         dfd.index = dfd.index.set_levels(np.positive(dfd.index.levels[ilev]), level=ilev)
                #         # dfd.cf["Z"] = np.positive(dfd.cf["Z"].values)
                #         # dfd[zkey] = np.positive(dfd[zkey].values)
                #                 # ilev = dfd.index.names.index(index.name)
                #                 # dfd.index = dfd.index.set_levels(index, level=ilev)

                # # if isinstance(dfd, (xr.DataArray, xr.Dataset)):
                # #     dfd.cf["Z"].attrs = attrs
                # #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                # #         dfd.cf["Z"].encoding = encoding
                # Z = dfd.cf["Z"].values
                # vertical_interp = True
                # logger.info(f"Will perform vertical interpolation, to depths {Z}.")

            # check for already-aligned model-data file
            # fname_aligned_orig: no info about time modifications
            # fname_aligned: fully specific name
            fname_aligned_orig = f"{cat.name}_{source_name}_{key_variable_data}"
            if pd.notnull(user_min_time) and pd.notnull(user_max_time):
                fname_aligned_orig = f"{fname_aligned_orig}_{str(user_min_time.date())}_{str(user_max_time.date())}"
            fname_aligned_orig = ALIGNED_CACHE_DIR(project_name) / fname_aligned_orig
            assert isinstance(fname_aligned_orig, pathlib.Path)
            # also for ts_mods
            fnamemods = ""
            if ts_mods is not None:
                for mod in ts_mods:
                    fnamemods += f"_{mod['name_mod']}"
            fname_aligned = fname_aligned_orig.with_name(
                fname_aligned_orig.stem + fnamemods
            ).with_suffix(fname_aligned_orig.suffix)

            if isinstance(dfd, pd.DataFrame):
                fname_aligned = fname_aligned.with_suffix(".csv")
            elif isinstance(dfd, xr.Dataset):
                fname_aligned = fname_aligned.with_suffix(".nc")
            else:
                raise TypeError("object is neither DataFrame nor Dataset.")
            logger.info(f"Aligned model-data file name is {fname_aligned}.")

            # use same file name as for aligned but with different path base and
            # make sure .nc
            model_file_name = (
                MODEL_CACHE_DIR(project_name) / fname_aligned_orig.stem
            ).with_suffix(".nc")
            logger.info(f"model file name is {model_file_name}.")
            if fname_aligned.is_file():
                logger.info(
                    "Reading previously-aligned model output and data for %s.",
                    source_name,
                )
                if isinstance(dfd, pd.DataFrame):
                    dd = pd.read_csv(fname_aligned)  # , parse_dates=True)

                    if "T" in dd.cf:
                        dd[dd.cf["T"].name] = pd.to_datetime(dd.cf["T"])

                    # assume all columns except last two are index columns
                    # last two should be obs and model
                    dd = dd.set_index(list(dd.columns[:-2]))
                elif isinstance(dfd, xr.Dataset):
                    dd = xr.open_dataset(fname_aligned)
                else:
                    raise TypeError("object is neither DataFrame nor Dataset.")
            else:

                # # # Combine and align the two time series of variable
                # # with cfp_set_options(custom_criteria=vocab.vocab):
                if isinstance(dfd, DataFrame) and key_variable_data not in dfd.cf:
                    msg = f"Key variable {key_variable_data} cannot be identified in dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue

                elif isinstance(
                    dfd, xr.DataArray
                ) and key_variable_data not in cf_xarray.accessor._get_custom_criteria(
                    dfd, key_variable_data, vocab.vocab
                ):
                    msg = f"Key variable {key_variable_data} cannot be identified in dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue

                # see if more than one column of data is being identified as key_variable_data
                # if more than one, log warning and then choose first
                if isinstance(dfd.cf[key_variable_data], DataFrame):
                    msg = f"More than one variable ({dfd.cf[key_variable_data].columns}) have been matched to input variable {key_variable_data}. The first {dfd.cf[key_variable_data].columns[0]} is being selected. To change this, modify the vocabulary so that the two variables are not both matched, or change the input data catalog."
                    logger.warning(msg)
                    # remove other data columns
                    for col in dfd.cf[key_variable_data].columns[1:]:
                        dfd.drop(col, axis=1, inplace=True)

                if isinstance(dfd, pd.DataFrame):
                    # ONLY DO THIS FOR DATAFRAMES
                    # dfd.cf["T"] = to_datetime(dfd.cf["T"])
                    # dfd.set_index(dfd.cf["T"], inplace=True)

                    # deal with possible time zone
                    if isinstance(dfd.index, pd.core.indexes.multi.MultiIndex):
                        index = dfd.index.get_level_values(dfd.cf["T"].name)
                    else:
                        index = dfd.index

                    if index.tz is not None:
                        logger.warning(
                            "Dataset %s had a timezone %s which is being removed. Make sure the timezone matches the model output.",
                            source_name,
                            str(index.tz),
                        )
                        # remove time zone
                        index = index.tz_convert(None)

                        if isinstance(dfd.index, pd.core.indexes.multi.MultiIndex):
                            # loop over levels in index so we know which level to replace
                            inds = []
                            for lev in range(dfd.index.nlevels):
                                ind = dfd.index.get_level_values(lev)
                                if dfd.index.names[lev] == dfd.cf["T"].name:
                                    ind = ind.tz_convert(None)
                                inds.append(ind)
                            dfd = dfd.set_index(inds)

                            # ilev = dfd.index.names.index(index.name)
                            # dfd.index = dfd.index.set_levels(index, level=ilev)
                            # # dfd.index.set_index([])
                        else:
                            dfd.index = index  # dfd.index.tz_convert(None)
                            dfd.cf["T"] = index  # dfd.index

                    # # make sure index is sorted ascending so time goes forward
                    # dfd = dfd.sort_index()

                logger.info(
                    "No previously-aligned model output and data available for %s, so setting up now.",
                    source_name,
                )

                # This is meant to limit the data range when user has input time range
                # for limiting time range of long datasets
                if (
                    pd.notnull(user_min_time)
                    and pd.notnull(user_max_time)
                    and (data_min_time.date() <= user_min_time.date())
                    and (data_max_time.date() >= user_max_time.date())
                ):
                    # if pd.notnull(user_min_time) and pd.notnull(user_max_time) and (abs(data_min_time - user_min_time) <= pd.Timedelta("1 day")) and (abs(data_max_time - user_max_time) >= pd.Timedelta("1 day")):
                    # if pd.notnull(user_min_time) and pd.notnull(user_max_time) and (data_min_time <= user_min_time) and (data_max_time >= user_max_time):
                    # if data_time_range.encompass(model_time_range):
                    dfd = dfd.loc[user_min_time:user_max_time]
                else:
                    dfd = dfd

                # check if all of variable is nan
                if dfd.cf[key_variable_data].isnull().all():
                    msg = f"All values of key variable {key_variable_data} are nan in dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    continue

                # Read in model output from cache if possible.

                # # use same file name as for aligned but with different path base and
                # # make sure .nc
                # model_file_name = (MODEL_CACHE_DIR(project_name) / fname_aligned.stem).with_suffix(".nc")

                # # create model file name
                # xkey, ykey, tkey = dam.cf['X'].name, dam.cf['Y'].name, dam.cf["T"].name
                # ix, iy = int(kwargs_out[xkey]), int(kwargs_out[ykey])
                # name = key_variable if isinstance(key_variable, str) else model_var.name
                # model_file_name = f"{name}_{xkey}_{ix}_{ykey}_{iy}"
                # # these two cases should be the same but for NWGOA Jan 1 is missing each year so it
                # # won't end up with the same dates
                # if pd.notnull(user_min_time) and pd.notnull(user_max_time):
                #     t0, t1 = str(user_min_time.date()), str(user_max_time.date())
                # else:
                #     t0, t1 = str(pd.Timestamp(dsm2.cf["T"][0].values).date()), str(pd.Timestamp(dsm2.cf["T"][-1].values).date())
                # model_file_name += f"_{tkey}_{t0}_{t1}"
                # if "Z" in dam.cf.axes:
                #     if vertical_interp:
                #         zkey = model_var.cf["Z"].name
                #         # make string from array of depth values
                #         zstr = f'_{zkey}_{"_".join(str(x) for x in Z)}'
                #         model_file_name += zstr
                #     else:
                #         zkey = dam.cf["Z"].name
                #         # iz = -1  # change this to s_rho value?
                #         iz = list(dam.cf["Z"].values).index(model_var[zkey].values)
                #         model_file_name += f"_{zkey}_{iz}"
                #     # indexer.update({zkey: iz})
                # model_file_name = (MODEL_CACHE_DIR(project_name) / model_file_name).with_suffix(".nc")
                # logger.info(f"model file name is {model_file_name}.")

                # # check length of file name for being too long
                # if len(model_file_name.stem) > 255:
                #     import hashlib
                #     m = hashlib.sha256(str(model_file_name.stem).encode('UTF-8'))
                #     new_stem = m.hexdigest()
                #     model_file_name = (model_file_name.parent / new_stem).with_suffix(".nc")
                #     logger.info(f"model file name is too long so using hash version {model_file_name}.")

                if model_file_name.is_file():
                    logger.info("Reading model output from file.")
                    model_var = xr.open_dataset(model_file_name)
                    # model_var = xr.open_dataarray(model_file_name)
                    if not interpolate_horizontal:
                        distance = model_var["distance"]
                    # try to help with missing attributes
                    model_var = model_var.cf.guess_coord_axis()
                    model_var = model_var.cf[key_variable_data]
                    # distance = model_var.attrs["distance_from_location_km"]

                    # if not model_file_name.is_file():
                    #     # dam.isel(indexer).cf.sel(T=slice(start_time, end_time)).to_netcdf(model_file_name)
                    #     logger.info(f"Saving model output to file.")
                    #     model_var = model_var.compute()
                    #     model_var.to_netcdf(model_file_name)

                    if model_only:
                        logger.info("Running model only so moving on to next source...")
                        continue

                # # to continue, read from file
                # else:
                #     logger.info("Reading model output from file.")
                #     model_var = xr.open_dataarray(model_file_name)

                # have to read in the model output
                else:

                    # logic for one or multiple lon/lat locations
                    if min_lon != max_lon or min_lat != max_lat:
                        logger.info(
                            f"Source {source_name} in catalog {cat.name} is not stationary so using multiple locations."
                        )
                        lons, lats = (
                            dfd.cf["longitude"].values,
                            dfd.cf["latitude"].values,
                        )
                    else:
                        lons, lats = min_lon, max_lat

                    ### MOVING MODEL TO HERE?
                    # put the initial connection earlier, so can check times, then this stuff here

                    # Do light preprocessing so that .cf["T"] will work
                    if preprocess and not preprocessed:

                        dsm = em.preprocess(dsm)
                        grid = None

                        # now has been preprocessed
                        preprocessed = True

                    # do not deal with time in detail here since that will happen when the model and data
                    # are "aligned" a little later. For now, just return a slice of model times, outside of the
                    # extract_model code since not interpolating yet.
                    # not dealing with case that data is available before or after model but overlapping
                    # rename dsm since it has fewer times now and might need them for the other datasets
                    if (
                        pd.notnull(user_min_time)
                        and pd.notnull(user_max_time)
                        and (model_min_time.date() <= user_min_time.date())
                        and (model_max_time.date() >= user_max_time.date())
                    ):
                        # if model_time_range.encompass(data_time_range):
                        dsm2 = dsm.cf.sel(T=slice(user_min_time, user_max_time))
                        # dsm2 = dsm.cf.sel(T=slice(pd.Timestamp(data_min_time) - pd.Timedelta("1 hour"),
                        #                          pd.Timestamp(data_max_time) + pd.Timedelta("1 hour")))
                    # elif data_min_time == data_max_time:
                    # always take an extra hour just in case
                    else:
                        dsm2 = dsm.cf.sel(
                            T=slice(
                                data_min_time - pd.Timedelta("1H"),
                                data_max_time + pd.Timedelta("1H"),
                            )
                        )
                    # else:
                    #     dsm2 = dsm.cf.sel(T=slice(data_min_time, data_max_time))

                    # process model output without using open_mfdataset
                    # vertical coords have been an issue for ROMS and POM, related to dask and OFS models
                    if preprocess and need_xgcm_grid:
                        # if em.preprocessing.guess_model_type(dsm) in ["ROMS", "POM"]:
                        #     kwargs_pp = {"interp_vertical": False}
                        # else:
                        #     kwargs_pp = {}
                        # dsm = em.preprocess(dsm, kwargs=kwargs_pp)

                        # if em.preprocessing.guess_model_type(dsm) in ["ROMS"]:
                        #     grid = em.preprocessing.preprocess_roms_grid(dsm)
                        # else:
                        #     grid = None
                        # dsm = em.preprocess(dsm, kwargs=dict(grid=grid))

                        if em.preprocessing.guess_model_type(dsm2) in ["ROMS"]:
                            if need_xgcm_grid:
                                import xroms

                                logger.info(
                                    "setting up for model output with xroms, might take a few minutes..."
                                )
                                kwargs_xroms = kwargs_xroms or {}
                                dsm2, grid = xroms.roms_dataset(dsm2, **kwargs_xroms)
                                dsm2.xroms.set_grid(grid)

                        # now has been preprocessed
                        preprocessed = True

                    # Calculate boundary of model domain to compare with data locations and for map
                    if p1 is None:
                        if not ALPHA_PATH(project_name).is_file():
                            # let it find a mask
                            _, _, _, p1 = find_bbox(
                                dsm,
                                alpha=alpha,
                                dd=dd,
                                save=True,
                                project_name=project_name,
                            )
                            logger.info("Calculating numerical domain boundary.")
                        else:
                            logger.info("Using existing numerical domain boundary.")
                            with open(ALPHA_PATH(project_name)) as f:
                                p1wkt = f.readlines()[0]
                            p1 = shapely.wkt.loads(p1wkt)

                    # see if data location is inside alphashape-calculated polygon of model domain
                    # This currently assumes that the dataset is fixed in space.
                    # BUT — might want to just use nearest point so make this optional
                    if check_in_boundary:
                        point = Point(min_lon, min_lat)
                        if not p1.contains(point):
                            msg = f"Dataset {source_name} at lon {min_lon}, lat {min_lat} not located within model domain. Skipping dataset.\n"
                            logger.warning(msg)
                            continue

                    # Narrow model from Dataset to DataArray here
                    # key_variable = ["xroms", "ualong", "theta"]  # and all necessary steps to get there will happen
                    # key_variable = {"accessor": "xroms", "function": "ualong", "inputs": {"theta": theta}}
                    # # HOW TO GET THETA IN THE DICT?

                    # dam might be a Dataset but it has to be on a single grid, that is, e.g., all variable on the ROMS rho grid.
                    # well, that is only partially true. em.select requires DataArrays for certain operations like vertical
                    # interpolation.
                    if isinstance(key_variable, dict):
                        # HAVE TO ADD ANGLE TO THE INPUTS HERE SOMEHOW
                        # check if we need to access anything from the dataset metadata in "add_to_inputs" entry
                        if "add_to_inputs" in key_variable:
                            new_input_val = cat[source_name].metadata[
                                list(key_variable["add_to_inputs"].values())[0]
                            ]
                            new_input_key = list(key_variable["add_to_inputs"].keys())[
                                0
                            ]
                            key_variable["inputs"].update(
                                {new_input_key: new_input_val}
                            )

                        # e.g. ds.xroms.east_rotated(angle=-90, reference="compass", isradians=False, name="along_channel")
                        dam = getattr(
                            getattr(dsm2, key_variable["accessor"]),
                            key_variable["function"],
                        )(**key_variable["inputs"])
                    else:

                        with cfx_set_options(custom_criteria=vocab.vocab):

                            dam = dsm2.cf[key_variable_data]

                            # # this is the case in which need to find the depth index
                            # # swap z_rho and z_rho0 in order to do this
                            # # doing this here since now we know the variable and have a DataArray
                            # if Z is not None and Z != 0 and not vertical_interp:

                            #     zkey = dam.cf["vertical"].name
                            #     zkey0 = f"{zkey}0"
                            #     if zkey0 not in dsm2.coords:
                            #         raise KeyError("missing time-invariant version of z coordinates.")
                            #     if zkey0 not in dam.coords:
                            #         dam[zkey0] = dsm[zkey0]
                            #         dam[zkey0].attrs = dam[zkey].attrs
                            #         dam = dam.drop(zkey)
                            #         if hasattr(dam, "encoding") and "coordinates" in dam.encoding:
                            #             dam.encoding["coordinates"] = dam.encoding["coordinates"].replace(zkey,zkey0)

                    # if dask-backed, read into memory
                    if dam.cf["longitude"].chunks is not None:
                        dam[dam.cf["longitude"].name] = dam.cf["longitude"].load()
                    if dam.cf["latitude"].chunks is not None:
                        dam[dam.cf["latitude"].name] = dam.cf["latitude"].load()

                    # shift if 0 to 360
                    dam = shift_longitudes(dam)  # this is fast if not needed

                    # expand 1D coordinates to 2D, so all models dealt with in OMSA are treated with 2D coords.
                    # if your model is too large to be treated with this way, subset the model first.
                    dam = coords1Dto2D(dam)  # this is fast if not needed

                    # take out relevant variable and identify mask if available (otherwise None)
                    # this mask has to match dam for em.select()
                    if mask is None:
                        if MASK_PATH(project_name, key_variable).is_file():
                            logger.info("Using cached mask.")
                            mask = xr.open_dataarray(
                                MASK_PATH(project_name, key_variable)
                            )
                        else:
                            logger.info("Finding and saving mask to cache.")
                            # # dam variable might not be in Dataset itself, but its coordinates probably are.
                            # mask = get_mask(dsm, dam.name)
                            mask = get_mask(
                                dsm, dam.cf["longitude"].name, wetdry=wetdry
                            )
                            mask.to_netcdf(MASK_PATH(project_name, key_variable))
                        # there should not be any nans in the mask!
                        if mask.isnull().any():
                            raise ValueError(
                                f"""there are nans in your mask — better fix something.
                                             The cached version is at {MASK_PATH(project_name, key_variable)}.
                                             """
                            )

                    # if vertical isn't present either the variable doesn't have the concept, like ssh, or it is missing
                    if "vertical" not in dam.cf.coordinates:
                        logger.warning(
                            "the 'vertical' key cannot be identified in dam by cf-xarray. Maybe you need to include the xgcm grid and vertical metrics for xgcm grid, but maybe your variable does not have a vertical axis."
                        )
                        # raise KeyError("the 'vertical' key cannot be identified in dam by cf-xarray. Maybe you need to include the xgcm grid and vertical metrics for xgcm grid.")

                    # # 1 location: interpolate or nearest neighbor horizontally
                    # # have it figure out depth
                    # if ("Z" not in dfd.cf.axes) or no_Z:
                    #     Z = None
                    #     vertical_interp = False
                    #     logger.info(f"Will not perform vertical interpolation and there is no concept of depth for {key_variable_data}.")
                    # elif (dfd.cf["Z"] == 0).all():
                    #     Z = 0  # do nearest depth to 0
                    #     vertical_interp = False
                    #     logger.info(f"Will not perform vertical interpolation and will find nearest depth to {Z}.")
                    # else:
                    #     # if the model depths are positive up/negative down, make sure the data match
                    #     attrs = dfd[dfd.cf["Z"].name].attrs
                    #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                    #         encoding = dfd[dfd.cf["Z"].name].encoding
                    #     zkey = dfd.cf["Z"].name
                    #     if dam.cf["vertical"].attrs["positive"] == "up":
                    #         dfd[zkey] = np.negative(dfd[zkey].values)
                    #     else:
                    #         dfd[zkey] = np.positive(dfd[zkey].values)
                    #     dfd[zkey].attrs = attrs
                    #     if hasattr(dfd[dfd.cf["Z"].name], "encoding"):
                    #         dfd[zkey].encoding = encoding
                    #     Z = dfd.cf["Z"].values
                    #     vertical_interp = True
                    #     logger.info(f"Will perform vertical interpolation, to depths {Z}.")

                    # use pickle of triangulation from project dir if available
                    tri_name = PROJ_DIR(project_name) / "tri.pickle"
                    if (
                        interpolate_horizontal
                        and horizontal_interp_code == "delaunay"
                        and tri_name.is_file()
                    ):
                        import pickle

                        logger.info(
                            f"Using previously-calculated Delaunay triangulation located at {tri_name}."
                        )

                        with open(tri_name, "rb") as handle:
                            tri = pickle.load(handle)
                    else:
                        tri = None

                    logger.info(
                        f"Selecting model output at locations to match dataset {source_name}."
                    )
                    model_var, kwargs_out = em.select(
                        # model_var, weights, distance, kwargs_out = em.select(
                        dam,
                        longitude=lons,
                        latitude=lats,
                        # T=slice(user_min_time, user_max_time),
                        T=dfd.cf["T"].values,
                        # T=None,  # changed this because wasn't working with CTD profiles. Time interpolation happens during _align.
                        make_time_series=True,  # advanced index to make result time series instead of array
                        Z=Z,
                        vertical_interp=vertical_interp,
                        iT=None,
                        iZ=None,
                        extrap=extrap,
                        extrap_val=None,
                        locstream=True,
                        # locstream_dim="z_rho",
                        weights=None,
                        mask=mask,
                        use_xoak=False,
                        horizontal_interp=interpolate_horizontal,
                        horizontal_interp_code=horizontal_interp_code,
                        triangulation=tri,
                        xgcm_grid=grid,
                        return_info=True,
                    )
                    # save pickle of triangulation to project dir
                    if (
                        interpolate_horizontal
                        and horizontal_interp_code == "delaunay"
                        and not tri_name.is_file()
                    ):
                        import pickle

                        with open(tri_name, "wb") as handle:
                            pickle.dump(
                                kwargs_out["tri"],
                                handle,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )

                    msg = f"""
                    Model coordinates found are {model_var.coords}.
                    """
                    if interpolate_horizontal:
                        msg += f"""
                    Interpolation coordinates used for horizontal interpolation are {kwargs_out["interp_coords"]}."""
                    else:
                        msg += f"""
                    Output information from finding nearest neighbors to requested points are {kwargs_out}."""
                    logger.info(msg)

                    # Use distances from xoak to give context to how far the returned model points might be from
                    # the data locations
                    if not interpolate_horizontal:
                        distance = kwargs_out["distances"]
                        if (distance > 5).any():
                            logger.warning(
                                "Distance between nearest model location and data location for source %s is over 5 km with a distance of %s",
                                source_name,
                                str(float(distance)),
                            )
                        elif (distance > 100).any():
                            msg = f"Distance between nearest model location and data location for source {source_name} is over 100 km with a distance of {float(distance)}. Skipping dataset.\n"
                            logger.warning(msg)
                            maps.pop(-1)
                            continue

                    if len(model_var.cf["T"]) == 0:
                        # model output isn't available to match data
                        # data must not be in the space/time range of model
                        maps.pop(-1)
                        logger.warning(
                            "Model output is not present to match dataset %s.",
                            source_name,
                        )
                        continue

                    # this is trying to drop z_rho type coordinates to not save an extra time series
                    if (
                        Z is not None
                        and not vertical_interp
                        and "vertical" in model_var.cf.coordinates
                    ):
                        logger.info("Trying to drop vertical coordinates time series")
                        model_var = model_var.drop_vars(model_var.cf["vertical"].name)

                    # try rechunking to avoid killing kernel
                    if model_var.dims == (model_var.cf["T"].name,):
                        # for simple case of only time, just rechunk into pieces if no chunks
                        if model_var.chunks == ((model_var.size,),):
                            logger.info(f"Rechunking model output...")
                            model_var = model_var.chunk({"ocean_time": 1})

                    logger.info(f"Loading model output...")
                    model_var = model_var.compute()
                    # depths shouldn't need to be saved if interpolated since then will be a dimension
                    if Z is not None and not vertical_interp:
                        # find Z index
                        if "Z" in dam.cf.axes:
                            zkey = dam.cf["Z"].name
                            iz = list(dam.cf["Z"].values).index(model_var[zkey].values)
                            model_var[f"i_{zkey}"] = iz
                        else:
                            raise KeyError("Z missing from dam axes")
                    if not interpolate_horizontal:
                        if len(distance) > 1:
                            model_var["distance"] = (
                                model_var.cf["T"].name,
                                distance,
                            )  # if more than one distance, it is array
                        else:
                            model_var["distance"] = float(distance)
                        model_var["distance"].attrs["units"] = "km"
                        # model_var.attrs["distance_from_location_km"] = float(distance)
                    else:
                        # when lons/lats are function of time, add them back in
                        if dam.cf["longitude"].name not in model_var.coords:
                            # if model_var.ndim == 1 and len(model_var[model_var.dims[0]]) == lons.size:
                            if isinstance(lons, (float, int)):
                                attrs = dict(
                                    axis="X",
                                    units="degrees_east",
                                    standard_name="longitude",
                                )
                                model_var[dam.cf["longitude"].name] = lons
                                model_var[dam.cf["longitude"].name].attrs = attrs
                            elif (
                                model_var.ndim == 1
                                and len(model_var[model_var.dims[0]]) == lons.size
                            ):
                                attrs = dict(
                                    axis="X",
                                    units="degrees_east",
                                    standard_name="longitude",
                                )
                                model_var[dam.cf["longitude"].name] = (
                                    model_var.dims[0],
                                    lons,
                                    attrs,
                                )
                        if dam.cf["latitude"].name not in model_var.dims:
                            if isinstance(lats, (float, int)):
                                model_var[dam.cf["latitude"].name] = lats
                                attrs = dict(
                                    axis="Y",
                                    units="degrees_north",
                                    standard_name="latitude",
                                )
                                model_var[dam.cf["latitude"].name].attrs = attrs
                            elif (
                                model_var.ndim == 1
                                and len(model_var[model_var.dims[0]]) == lats.size
                            ):
                                attrs = dict(
                                    axis="Y",
                                    units="degrees_north",
                                    standard_name="latitude",
                                )
                                model_var[dam.cf["latitude"].name] = (
                                    model_var.dims[0],
                                    lats,
                                    attrs,
                                )
                    attrs = {
                        "key_variable": key_variable,
                        "vertical_interp": str(vertical_interp),
                        "interpolate_horizontal": str(interpolate_horizontal),
                        "model_source_name": model_source_name,
                        "source_name": source_name,
                    }
                    if interpolate_horizontal:
                        attrs.update(
                            {
                                "horizontal_interp_code": horizontal_interp_code,
                            }
                        )
                    model_var.attrs.update(attrs)

                    logger.info(f"Saving model output to file...")
                    model_var.to_netcdf(model_file_name)

                if model_only:
                    logger.info("Running model only so moving on to next source...")
                    continue

                # # to continue, read from file
                # else:
                #     logger.info("Reading model output from file.")
                #     model_var = xr.open_dataarray(model_file_name)

                # # this should be in extract_model or future xoceanmodel instead of here directly
                # if tidal_filtering is not None:
                #     import oceans.filters
                #     if "data" in tidal_filtering and tidal_filtering["data"]:
                #         raise NotImplementedError()
                #         # dfd.cf[key_variable_data] = dfd.cf[key_variable_data]
                #     elif "model" in tidal_filtering and tidal_filtering["model"]:
                #         # logger.info(f"Loading in selected model output.")
                #         # model_var = model_var.compute()
                #         logger.info(f"Tidally filtering model output.")
                #         model_var = oceans.filters.pl33tn(model_var)

                # opportunity to modify time series data
                # fnamemods = ""
                if ts_mods is not None:
                    for mod in ts_mods:
                        logger.info(
                            f"Apply a time series modification called {mod['function']}."
                        )
                        dfd[dfd.cf[key_variable_data].name] = mod["function"](
                            dfd.cf[key_variable_data], **mod["inputs"]
                        )
                        model_var = mod["function"](model_var, **mod["inputs"])
                        # fnamemods += f"_{mod['name_mod']}"
                # fname_aligned = fname_aligned.with_name(fname_aligned.stem + fnamemods).with_suffix(fname_aligned.suffix)
                # fname_aligned = fname_aligned.with_name(fname_aligned.stem + f"_{mod['name_mod']}").with_suffix(fname_aligned.suffix)

                logger.info(
                    "Aligning model output and data for %s.",
                    source_name,
                )
                # input all context dimensions
                # cols = ["Z","T","longitude","latitude"]
                # varnames = [dfd.cf.axes[col][0] for col in cols if col in dfd.cf.axes]
                # varnames += [dfd.cf.coordinates[col][0] for col in cols if col in dfd.cf.coordinates]
                # varnames += [dfd.cf[key_variable_data].name]
                # dd = _align(dfd[varnames], model_var, key_variable=key_variable_data)
                dd = _align(dfd.cf[key_variable_data], model_var)
                # read in from newly made file to make sure output is loaded
                if isinstance(dd, pd.DataFrame):
                    dd.to_csv(fname_aligned)
                    dd = pd.read_csv(fname_aligned, index_col=0, parse_dates=True)
                elif isinstance(dd, xr.Dataset):
                    dd.to_netcdf(fname_aligned)
                    dd = xr.open_dataset(fname_aligned)
                else:
                    raise TypeError("object is neither DataFrame nor Dataset.")
                # y_name = model_var.name

            # model_file_name = (MODEL_CACHE_DIR(project_name) / fname_aligned.stem).with_suffix(".nc")
            logger.info(f"model file name is {model_file_name}.")
            if model_file_name.is_file():
                logger.info("Reading model output from file.")
                model_var = xr.open_dataset(model_file_name)
                if not interpolate_horizontal:
                    distance = model_var["distance"]
                # distance = model_var.attrs["distance_from_location_km"]
            else:
                raise ValueError("If the aligned file is available need this one too.")

            stats_fname = (OUT_DIR(project_name) / f"{fname_aligned.stem}").with_suffix(
                ".yaml"
            )
            # stats_fname = OUT_DIR(project_name) / f"stats_{source_name}_{key_variable_data}.yaml"

            if stats_fname.is_file():
                logger.info("Reading from previously-saved stats file.")
                with open(stats_fname, "r") as stream:
                    stats = yaml.safe_load(stream)

            else:
                # Where to save stats to?
                stats = dd.omsa.compute_stats

                # add distance in
                if not interpolate_horizontal:
                    stats["dist"] = float(distance)
                    # stats["dist"] = float(distance)

                # save stats
                # stats_file_name = f"stats_{source_name}_{key_variable}"
                # if pd.notnull(user_min_time) and pd.notnull(user_max_time):
                #     stats_file_name = f"{stats_file_name}_{str(user_min_time.date())}_{str(user_max_time.date())}"
                # stats_file_name = (OUT_DIR(project_name) / stats_file_name).with_suffix(".yaml")

                save_stats(
                    source_name,
                    stats,
                    project_name,
                    key_variable_data,
                    filename=stats_fname,
                )
                logger.info("Saved stats file.")

            # Write stats on plot
            figname = (OUT_DIR(project_name) / f"{fname_aligned.stem}").with_suffix(
                ".png"
            )

            # figname = f"{source_name}_{key_variable_data}"
            # if pd.notnull(user_min_time) and pd.notnull(user_max_time):
            #     figname = f"{figname}_{str(user_min_time.date())}_{str(user_max_time.date())}"
            # figname = (OUT_DIR(project_name) / figname).with_suffix(".png")
            if plot_count_title:
                title = f"{count}: {source_name}"
            else:
                title = f"{source_name}"
            dd.omsa.plot(
                title=title,
                key_variable=key_variable,
                # ylabel=key_variable,
                figname=figname,
                stats=stats,
                featuretype=cat[source_name].metadata["featuretype"],
                cmap="cmo.delta",
                clabel=key_variable,
            )
            msg = f"Plotted time series for {source_name}\n."
            logger.info(msg)

            count += 1

    # map of model domain with data locations
    if plot_map:
        if len(maps) > 0:
            try:
                figname = OUT_DIR(project_name) / "map.png"
                map.plot_map(np.asarray(maps), figname, p=p1, **kwargs_map)
            except ModuleNotFoundError:
                pass
        else:
            logger.warning("Not plotting map since no datasets to plot.")
    logger.info(
        "Finished analysis. Find plots, stats summaries, and log in %s.",
        str(PROJ_DIR(project_name)),
    )
    # logger.shutdown()
    # logging.shutdown()
