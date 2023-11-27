"""
Main run functions.
"""

import logging
import mimetypes
import pathlib
import warnings

from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple, Union

import cf_xarray
import extract_model as em
import extract_model.accessor
import intake
import matplotlib.pyplot as plt
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
from xgcm import Grid

# from ocean_model_skill_assessor.plot import map
import ocean_model_skill_assessor.plot as plot

from .featuretype import ftconfig
from .paths import Paths
from .stats import compute_stats, save_stats
from .utils import (
    check_catalog,
    check_dataframe,
    check_dataset,
    coords1Dto2D,
    find_bbox,
    fix_dataset,
    get_mask,
    kwargs_search_from_model,
    open_catalogs,
    open_vocab_labels,
    open_vocabs,
    read_model_file,
    read_processed_data_file,
    save_processed_files,
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
    skip_strings: Optional[list] = None,
    kwargs_open: Optional[Dict] = None,
    logger=None,
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
    skip_entry_metadata : bool, optional
        This is useful for testing in which case we don't want to actually read the file. If you are making a catalog file for a model, you may want to set this to `True` to avoid reading it all in for metadata.
    skip_strings : list of strings, optional
        If provided, source_names in catalog will only be checked for goodness if they do not contain one of skip_strings. For example, if `skip_strings=["_base"]` then any source in the catalog whose name contains that string will be skipped.
    kwargs_open : dict, optional
        Keyword arguments to pass on to the appropriate ``intake`` ``open_*`` call for model or dataset.

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
            if isinstance(dd, pd.DataFrame):
                dd[dd.cf["T"].name] = to_datetime(dd.cf["T"])
                dd.set_index(dd.cf["T"], inplace=True)
                if dd.index.tz is not None:
                    # logger is already defined in other function
                    if logger is not None:
                        logger.warning(  # type: ignore
                            "Dataset %s had a timezone %s which is being removed. Make sure the timezone matches the model output.",
                            source,
                            str(dd.index.tz),
                        )
                    dd.index = dd.index.tz_convert(None)
                    dd.cf["T"] = dd.index

            metadata.update(
                {
                    "minTime": str(dd.cf["T"].values.min()),  # works for df and ds!
                    "maxTime": str(dd.cf["T"].values.max()),  # works for df and ds!
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

    # this allows for not checking a model catalog
    if not skip_entry_metadata:
        check_catalog(cat, skip_strings=skip_strings)

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
    skip_strings: Optional[list] = None,
    vocab: Optional[Union[Vocab, str, PurePath]] = None,
    return_cat: bool = True,
    save_cat: bool = False,
    verbose: bool = True,
    mode: str = "w",
    testing: bool = False,
    cache_dir: Optional[Union[str, PurePath]] = None,
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
    skip_strings : list of strings, optional
        If provided, source_names in catalog will only be checked for goodness if they do not contain one of skip_strings. For example, if `skip_strings=["_base"]` then any source in the catalog whose name contains that string will be skipped.
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
    cache_dir: str, Path
        Pass on to omsa.paths to set cache directory location if you don't want to use the default. Good for testing.
    """

    paths = Paths(project_name, cache_dir=cache_dir)

    logger = set_up_logging(verbose, paths=paths, mode=mode, testing=testing)

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
        kwargs_search = kwargs_search_from_model(kwargs_search, paths)

    if vocab is not None:
        if isinstance(vocab, str):
            vocab = Vocab(paths.VOCAB_PATH(vocab))
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
            skip_strings=skip_strings,
            logger=logger,
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

    # this allows for not checking a model catalog
    if "skip_entry_metadata" in kwargs and not kwargs["skip_entry_metadata"]:
        check_catalog(cat, skip_strings=skip_strings)

    if save_cat:
        # save cat to file
        cat.save(paths.CAT_PATH(catalog_name))
        logger.info(
            f"Catalog saved to {paths.CAT_PATH(catalog_name)} with {len(list(cat))} entries."
        )

    if return_cat:
        return cat


def _initial_model_handling(
    model_name: Union[str, Catalog],
    paths: Paths,
    model_source_name: Optional[str] = None,
) -> xr.Dataset:
    """Initial model handling.

    cf-xarray needs to be able to identify Z, T, longitude, latitude coming out of here.

    Parameters
    ----------
    model_name : str, Catalog
        Name of catalog for model output, created with ``make_catalog`` call, or Catalog instance.
    paths : Paths
        Paths object for finding paths to use.
    model_source_name : str, optional
        Use this to access a specific source in the input model_catalog instead of otherwise just using the first source in the catalog.

    Returns
    -------
    Dataset
        Dataset pointing to model output.
    """

    # read in model output
    model_cat = open_catalogs(model_name, paths, skip_check=True)[0]
    model_source_name = model_source_name or list(model_cat)[0]
    dsm = model_cat[model_source_name].to_dask()

    # the main preprocessing happens later, but do a minimal job here
    # so that cf-xarray can be used hopefully
    dsm = em.preprocess(dsm, kwargs=dict(find_depth_coords=False))

    check_dataset(dsm)
    return dsm, model_source_name


def _narrow_model_time_range(
    dsm: xr.Dataset,
    user_min_time: pd.Timestamp,
    user_max_time: pd.Timestamp,
    model_min_time: pd.Timestamp,
    model_max_time: pd.Timestamp,
    data_min_time: pd.Timestamp,
    data_max_time: pd.Timestamp,
) -> xr.Dataset:
    """Narrow the model time range to approximately what is needed, to save memory.

    If user_min_time and user_max_time were input and are not null values and are narrower than the model time range, use those to control time range.

    Otherwise use data_min_time and data_max_time to narrow the time range, but add 1 model timestep on either end to make sure to have extra model output if need to interpolate in that range.

    Do not deal with time in detail here since that will happen when the model and data
    are "aligned" a little later. For now, just return a slice of model times, outside of the
    extract_model code since not interpolating yet.
    not dealing with case that data is available before or after model but overlapping
    rename dsm since it has fewer times now and might need them for the other datasets

    Parameters
    ----------
    dsm: xr.Dataset
        model dataset
    user_min_time : pd.Timestamp
        If this is input, it will be used as the min time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    user_max_time : pd.Timestamp
        If this is input, it will be used as the max time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    model_min_time : pd.Timestamp
        Min model time step
    model_max_time : pd.Timestamp
        Max model time step
    data_min_time : pd.Timestamp
        The min time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_min_time, then the constraint time.
    data_max_time : pd.Timestamp
        The max time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_max_time, then the constraint time.

    Returns
    -------
    xr.Dataset
        Model dataset, but narrowed in time.
    """

    # calculate delta time for model
    dt = pd.Timestamp(dsm.cf["T"][1].values) - pd.Timestamp(dsm.cf["T"][0].values)

    if (
        pd.notnull(user_min_time)
        and pd.notnull(user_max_time)
        and (model_min_time.date() <= user_min_time.date())
        and (model_max_time.date() >= user_max_time.date())
    ):
        dsm2 = dsm.cf.sel(T=slice(user_min_time, user_max_time))

    # always take an extra timestep just in case
    else:
        dsm2 = dsm.cf.sel(
            T=slice(
                data_min_time - dt,
                data_max_time + dt,
            )
        )

    return dsm2


def _find_data_time_range(cat: Catalog, source_name: str) -> tuple:
    """Determine min and max data times.

    Parameters
    ----------
    cat : Catalog
        Catalog that contains dataset source_name from which to find data time range.
    source_name : str
        Name of dataset within cat to examine.

    Returns
    -------
    data_min_time : pd.Timestamp
        The min time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_min_time, then the constraint time. If "Z" is present to indicate UTC timezone, it is removed.
    data_max_time : pd.Timestamp
        The max time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_max_time, then the constraint time.  If "Z" is present to indicate UTC timezone, it is removed.
    """

    # Do min and max separately.
    if "minTime" in cat[source_name].metadata:
        data_min_time = cat[source_name].metadata["minTime"]
    # use kwargs_search min/max times if available
    elif (
        "kwargs_search" in cat.metadata and "min_time" in cat.metadata["kwargs_search"]
    ):
        data_min_time = cat.metadata["kwargs_search"]["min_time"]
    else:
        raise KeyError("Need a way to input min time desired.")

    # max
    if "maxTime" in cat[source_name].metadata:
        data_max_time = cat[source_name].metadata["maxTime"]
    # use kwargs_search min/max times if available
    elif (
        "kwargs_search" in cat.metadata and "max_time" in cat.metadata["kwargs_search"]
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

    return data_min_time, data_max_time


def _choose_depths(
    dd: Union[pd.DataFrame, xr.Dataset],
    model_depth_attr_positive: str,
    no_Z: bool,
    want_vertical_interp: bool,
    logger=None,
) -> tuple:
    """Determine depths to interpolate to, if any.

    This assumes the data container does not have indices, or at least no depth indices.

    Parameters
    ----------
    dd: DataFrame or Dataset
        Data container
    model_depth_attr_positive: str
        result of model.cf["Z"].attrs["positive"]: "up" or "down", from model
    no_Z : bool
        If True, set Z=None so no vertical interpolation or selection occurs. Do this if your variable has no concept of depth, like the sea surface height.
    want_vertical_interp: bool
        This is False unless the user wants to specify that vertical interpolation should happen. This is used in only certain cases but in those cases it is important so that it is known to interpolate instead of try to figure out a vertical level index (which is not possible currently).
    logger : logger, optional
        Logger for messages.

    Returns
    -------
    dd
        Possibly modified Dataset with sign of depths to match model
    Z
        Depths to interpolate to with sign that matches the model depths.
    vertical_interp
        Flag, True if we should interpolate vertically, False if not.
    """

    # sort out depths between model and data
    # 1 location: interpolate or nearest neighbor horizontally
    # have it figure out depth
    if ("Z" not in dd.cf.axes) or no_Z:
        Z = None
        vertical_interp = False
        if logger is not None:
            logger.info(
                f"Will not perform vertical interpolation and there is no concept of depth for this variable."
            )

    elif (dd.cf["Z"].size == 1) or (dd.cf["Z"] == dd.cf["Z"][0]).all():
        if dd.cf["Z"].size == 1:
            Z = float(dd.cf["Z"])
        else:
            Z = float(
                dd.cf["Z"][0]
            )  # do nearest depth to the one depth represented in dataset
        vertical_interp = False
        if logger is not None:
            logger.info(
                f"Will not perform vertical interpolation and will find nearest depth to {Z}."
            )

    # if depth varies in time and will interpolate to match depths
    elif (dd.cf["Z"] != dd.cf["Z"][0]).any() and want_vertical_interp:

        # if the model depths are positive up/negative down, make sure the data match
        if isinstance(dd, (xr.DataArray, xr.Dataset)):
            attrs = dd[dd.cf["Z"].name].attrs
            if hasattr(dd[dd.cf["Z"].name], "encoding"):
                encoding = dd[dd.cf["Z"].name].encoding

            if model_depth_attr_positive == "up":
                dd[dd.cf["Z"].name] = np.negative(dd.cf["Z"])
            else:
                dd[dd.cf["Z"].name] = np.positive(dd.cf["Z"])

            dd.cf["Z"].attrs = attrs
            if hasattr(dd[dd.cf["Z"].name], "encoding"):
                dd.cf["Z"].encoding = encoding

        elif isinstance(dd, (pd.DataFrame, pd.Series)):
            if model_depth_attr_positive == "up":
                dd.cf["Z"] = np.negative(abs(dd.cf["Z"]))
            else:
                dd.cf["Z"] = np.positive(abs(dd.cf["Z"]))

        Z = dd.cf["Z"].values
        vertical_interp = True

        if logger is not None:
            logger.info(f"Will perform vertical interpolation, to depths {Z}.")

    # if depth varies in time and need to determine depth index
    else:
        raise NotImplementedError(
            "Method to find index for depth not at surface not available yet."
        )

    return dd, Z, vertical_interp


def _dam_from_dsm(
    dsm2: xr.Dataset,
    key_variable: Union[str, dict],
    key_variable_data: str,
    source_metadata: dict,
    no_Z: bool,
    logger=None,
) -> xr.DataArray:
    """Select or calculate variable from Dataset.

    cf-xarray needs to work for Z, T, longitude, latitude after this

    Parameters
    ----------
    dsm2 : Dataset
        Dataset containing model output. If this is being run from `main`, the model output has already been narrowed to the relevant time range.
    key_variable : str, dict
        Information to select variable from Dataset. Will be a dict if something needs to be calculated or accessed. In the more simple case will be a string containing the key variable name that can be interpreted with cf-xarray to access the variable of interest from the Dataset.
    key_variable_data : str
        A string containing the key variable name that can be interpreted with cf-xarray to access the variable of interest from the Dataset.
    source_metadata : dict
        Metadata for dataset source. Accessed by `cat[source_name].metadata`.
    no_Z : bool
        If True, set Z=None so no vertical interpolation or selection occurs. Do this if your variable has no concept of depth, like the sea surface height.
    logger : logger, optional
        Logger for messages.

    Returns
    -------
    DataArray:
        Single variable DataArray from Dataset.
    """

    if isinstance(key_variable, dict):
        # HAVE TO ADD ANGLE TO THE INPUTS HERE SOMEHOW
        # check if we need to access anything from the dataset metadata in "add_to_inputs" entry
        if "add_to_inputs" in key_variable:
            new_input_val = source_metadata[
                list(key_variable["add_to_inputs"].values())[0]
            ]
            new_input_key = list(key_variable["add_to_inputs"].keys())[0]
            key_variable["inputs"].update({new_input_key: new_input_val})

        # e.g. ds.xroms.east_rotated(angle=-90, reference="compass", isradians=False, name="along_channel")
        function_or_property = getattr(
            getattr(dsm2, key_variable["accessor"]),
            key_variable["function"],
        )
        # if it is a property can't call it like a function
        if isinstance(getattr(type(dsm2.xroms), "east"), property):
            dam = function_or_property
        else:
            dam = function_or_property(**key_variable["inputs"])
    else:
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

    check_dataset(dam, no_Z=no_Z)

    # if dask-backed, read into memory
    if dam.cf["longitude"].chunks is not None:
        dam[dam.cf["longitude"].name] = dam.cf["longitude"].load()
    if dam.cf["latitude"].chunks is not None:
        dam[dam.cf["latitude"].name] = dam.cf["latitude"].load()

    # if vertical isn't present either the variable doesn't have the concept, like ssh, or it is missing
    if "Z" not in dam.cf.coordinates:
        if logger is not None:
            logger.warning(
                "the 'vertical' key cannot be identified in dam by cf-xarray. Maybe you need to include the xgcm grid and vertical metrics for xgcm grid, but maybe your variable does not have a vertical axis."
            )
        # raise KeyError("the 'vertical' key cannot be identified in dam by cf-xarray. Maybe you need to include the xgcm grid and vertical metrics for xgcm grid.")

    return dam


def _processed_file_names(
    fname_processed_orig: Union[str, pathlib.Path],
    dfd_type: type,
    user_min_time: pd.Timestamp,
    user_max_time: pd.Timestamp,
    paths: Paths,
    ts_mods: list,
    logger=None,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Determine file names for base of stats and figure names and processed data and model names

    fname_processed_orig: no info about time modifications
    fname_processed: fully specific name
    fname_processed_data: processed data file
    fname_processed_model: processed model file

    Parameters
    ----------
    fname_processed_orig : str
        Filename based but without modification if user_min_time and user_max_time were input. Does include info about ts_mods if present.
    dfd_type : type
        pd.DataFrame or xr.Dataset depending on the data container type.
    user_min_time : pd.Timestamp
        If this is input, it will be used as the min time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    user_max_time : pd.Timestamp
        If this is input, it will be used as the max time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    paths : Paths
        Paths object for finding paths to use.
    ts_mods : list
        list of time series modifications to apply to data and model. Can be an empty list if no modifications to apply.
    logger : logger, optional
        Logger for messages.

    Returns
    -------
    tuple of Paths
        * fname_processed: base to be used for stats and figure
        * fname_processed_data: file name for processed data
        * fname_processed_model: file name for processed model
        * model_file_name: (unprocessed) model output
    """

    if pd.notnull(user_min_time) and pd.notnull(user_max_time):
        fname_processed_orig = f"{fname_processed_orig}_{str(user_min_time.date())}_{str(user_max_time.date())}"
    fname_processed_orig = paths.PROCESSED_CACHE_DIR / fname_processed_orig
    assert isinstance(fname_processed_orig, pathlib.Path)

    # also for ts_mods
    fnamemods = ""
    for mod in ts_mods:
        fnamemods += f"_{mod['name_mod']}"
    fname_processed = fname_processed_orig.with_name(
        fname_processed_orig.stem + fnamemods
    ).with_suffix(fname_processed_orig.suffix)

    if dfd_type == pd.DataFrame:
        fname_processed_data = (
            fname_processed.parent / (fname_processed.stem + "_data")
        ).with_suffix(".csv")
    elif dfd_type == xr.Dataset:
        fname_processed_data = (
            fname_processed.parent / (fname_processed.stem + "_data")
        ).with_suffix(".nc")
    else:
        raise TypeError("object is neither DataFrame nor Dataset.")

    fname_processed_model = (
        fname_processed.parent / (fname_processed.stem + "_model")
    ).with_suffix(".nc")

    # use same file name as for processed but with different path base and
    # make sure .nc
    model_file_name: pathlib.Path = (
        paths.MODEL_CACHE_DIR / fname_processed_orig.stem
    ).with_suffix(".nc")

    if logger is not None:
        logger.info(f"Processed data file name is {fname_processed_data}.")
        logger.info(f"Processed model file name is {fname_processed_model}.")
        logger.info(f"model file name is {model_file_name}.")

    return fname_processed, fname_processed_data, fname_processed_model, model_file_name


def _check_prep_narrow_data(
    dd: Union[pd.DataFrame, xr.Dataset],
    key_variable_data: str,
    source_name: str,
    maps: list,
    vocab: Vocab,
    user_min_time: pd.Timestamp,
    user_max_time: pd.Timestamp,
    data_min_time: pd.Timestamp,
    data_max_time: pd.Timestamp,
    logger=None,
) -> Tuple[Union[pd.DataFrame, xr.Dataset], list]:
    """Check, prep, and narrow the data in time range.

    Parameters
    ----------
    dd : Union[pd.DataFrame, xr.Dataset]
        Dataset.
    key_variable_data : str
        Name of variable to access from dataset.
    source_name : str
        Name of dataset we are accessing from the catalog.
    maps : list
        Each entry is a list of information about a dataset; the last entry is for the present source_name or dataset. Each entry contains [min_lon, max_lon, min_lat, max_lat, source_name] and possibly an additional element containing "maptype".
    vocab : Vocab
        Way to find the criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for.
    user_min_time : pd.Timestamp
        If this is input, it will be used as the min time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    user_max_time : pd.Timestamp
        If this is input, it will be used as the max time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    data_min_time : pd.Timestamp
        The min time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_min_time, then the constraint time.
    data_max_time : pd.Timestamp
        The max time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_max_time, then the constraint time.
    logger : optional
        logger, by default None

    Returns
    -------
    tuple
        * dd: data container that has been checked and processed. Will be None if a problem has been detected.
        * maps: list of data information. If there was a problem with this dataset, the final entry in `maps` representing the dataset will have been deleted.
    """

    if isinstance(dd, DataFrame) and key_variable_data not in dd.cf:
        msg = f"Key variable {key_variable_data} cannot be identified in dataset {source_name}. Skipping dataset.\n"
        logger.warning(msg)
        maps.pop(-1)
        return None, maps

    elif (
        isinstance(dd, xr.DataArray)
        and vocab is not None
        and key_variable_data
        not in cf_xarray.accessor._get_custom_criteria(
            dd, key_variable_data, vocab.vocab
        )
    ):
        msg = f"Key variable {key_variable_data} cannot be identified in dataset {source_name}. Skipping dataset.\n"
        logger.warning(msg)
        maps.pop(-1)
        return None, maps

    # see if more than one column of data is being identified as key_variable_data
    # if more than one, log warning and then choose first
    # variable might be calculated later
    if key_variable_data in dd.cf and isinstance(dd.cf[key_variable_data], DataFrame):
        msg = f"More than one variable ({dd.cf[key_variable_data].columns}) have been matched to input variable {key_variable_data}. The first {dd.cf[key_variable_data].columns[0]} is being selected. To change this, modify the vocabulary so that the two variables are not both matched, or change the input data catalog."
        logger.warning(msg)
        # remove other data columns
        for col in dd.cf[key_variable_data].columns[1:]:
            dd.drop(col, axis=1, inplace=True)

    if isinstance(dd, pd.DataFrame):

        # shouldn't need to deal with multi-indices anymore
        # deal with possible time zone
        # if isinstance(dd.index, pd.core.indexes.multi.MultiIndex):
        #     index = dd.index.get_level_values(dd.cf["T"].name)
        # else:
        #     index = dd.index

        # if hasattr(index, "tz") and index.tz is not None:
        if dd.cf["T"].dt.tz is not None:
            logger.warning(
                "Dataset %s had a timezone %s which is being removed. Make sure the timezone matches the model output.",
                source_name,
                str(dd.cf["T"].dt.tz),
            )
            # remove time zone
            dd.cf["T"] = dd.cf["T"].dt.tz_convert(None)

            # if isinstance(dd.index, pd.core.indexes.multi.MultiIndex):
            #     # loop over levels in index so we know which level to replace
            #     inds = []
            #     for lev in range(dd.index.nlevels):
            #         ind = dd.index.get_level_values(lev)
            #         if dd.index.names[lev] == dd.cf["T"].name:
            #             ind = ind.tz_convert(None)
            #         inds.append(ind)
            #     dd = dd.set_index(inds)

            #     # ilev = dd.index.names.index(index.name)
            #     # dd.index = dd.index.set_levels(index, level=ilev)
            #     # # dd.index.set_index([])
            # else:
            #     dd.index = index  # dd.index.tz_convert(None)
            #     dd.cf["T"] = index  # dd.index

        # # make sure index is sorted ascending so time goes forward
        # dd = dd.sort_index()
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
        dd = (
            dd.set_index(dd.cf["T"])
            .loc[user_min_time:user_max_time]
            .reset_index(drop=True)
        )
    else:
        dd = dd

    # check if all of variable is nan
    # variable might be calculated later
    if key_variable_data in dd.cf and dd.cf[key_variable_data].isnull().all():
        msg = f"All values of key variable {key_variable_data} are nan in dataset {source_name}. Skipping dataset.\n"
        logger.warning(msg)
        maps.pop(-1)
        return None, maps

    return dd, maps


def _check_time_ranges(
    source_name: str,
    data_min_time: pd.Timestamp,
    data_max_time: pd.Timestamp,
    model_min_time: pd.Timestamp,
    model_max_time: pd.Timestamp,
    user_min_time: pd.Timestamp,
    user_max_time: pd.Timestamp,
    maps,
    logger=None,
) -> Tuple[bool, list]:
    """Compare time ranges in case should skip dataset source_name.

    Parameters
    ----------
    source_name : str
        Name of dataset we are accessing from the catalog.
    data_min_time : pd.Timestamp
        The min time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_min_time, then the constraint time.
    data_max_time : pd.Timestamp
        The max time in the dataset catalog metadata, or if there is a constraint in the metadata such as  an ERDDAP catalog allows, and it is more constrained than data_max_time, then the constraint time.
    user_min_time : pd.Timestamp
        If this is input, it will be used as the min time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    user_max_time : pd.Timestamp
        If this is input, it will be used as the max time for the model. At this point in the code, it will  be a pandas Timestamp though could be "NaT" (a null time value).
    model_min_time : pd.Timestamp
        Min model time step
    model_max_time : pd.Timestamp
        Max model time step
    maps : list
        Each entry is a list of information about a dataset; the last entry is for the present source_name or dataset. Each entry contains [min_lon, max_lon, min_lat, max_lat, source_name] and possibly an additional element containing "maptype".
    logger : logger, optional
        Logger for messages.

    Returns
    -------
    tuple
        * skip_dataset: bool that is True if this dataset should be skipped
        * maps: list of dataset information with the final entry (representing the present dataset) removed if skip_dataset is True.
    """

    if logger is not None:
        min_lon, max_lon, min_lat, max_lat = maps[-1][:4]
        logger.info(
            f"""
                        User time range: {user_min_time} to {user_max_time}.
                        Model time range: {model_min_time} to {model_max_time}.
                        Data time range: {data_min_time} to {data_max_time}.
                        Data lon range: {min_lon} to {max_lon}.
                        Data lat range: {min_lat} to {max_lat}."""
        )

    data_time_range = DateTimeRange(data_min_time, data_max_time)
    model_time_range = DateTimeRange(model_min_time, model_max_time)
    user_time_range = DateTimeRange(user_min_time, user_max_time)

    if not data_time_range.is_intersection(model_time_range):
        msg = f"Time range of dataset {source_name} and model output do not overlap. Skipping dataset.\n"
        if logger is not None:
            logger.warning(msg)
        maps.pop(-1)
        return True, maps

    if (
        pd.notnull(user_min_time)
        and pd.notnull(user_max_time)
        and not data_time_range.is_intersection(user_time_range)
    ):
        msg = f"Time range of dataset {source_name} and user-input time range do not overlap. Skipping dataset.\n"
        if logger is not None:
            logger.warning(msg)
        maps.pop(-1)
        return True, maps

    # in certain cases, the user input time range might be outside of the model availability
    if (
        pd.notnull(user_min_time)
        and pd.notnull(user_max_time)
        and not model_time_range.is_intersection(user_time_range)
    ):
        if logger is not None:
            logger.warning(
                "User-input time range is outside of model availability, so moving on..."
            )
        return True, maps
    return False, maps


def _return_p1(
    paths: Paths,
    dsm: xr.Dataset,
    mask: Union[xr.DataArray, None],
    alpha: int,
    dd: int,
    logger=None,
) -> shapely.Polygon:
    """Find and return the model domain boundary.

    Parameters
    ----------
    paths : Paths
        _description_
    dsm : xr.Dataset
        _description_
    mask : xr.DataArray or None
        Values are 1 for active cells and 0 for inactive grid cells in the model dsm.
    alpha: int, optional
        Number for alphashape to determine what counts as the convex hull. Larger number is more detailed, 1 is a good starting point.
    dd: int, optional
        Number to decimate model output lon/lat, as a stride.
    skip_mask : bool
        Allows user to override mask behavior and keep it as None. Good for testing. Default False.
    logger : _type_, optional
        _description_, by default None

    Returns
    -------
    shapely.Polygon
        Model domain boundary
    """

    if not paths.ALPHA_PATH.is_file():
        # let it find a mask
        _, _, _, p1 = find_bbox(
            dsm,
            paths=paths,
            mask=mask,
            alpha=alpha,
            dd=dd,
            save=True,
        )
        if logger is not None:
            logger.info("Calculating numerical domain boundary.")
    else:
        if logger is not None:
            logger.info("Using existing numerical domain boundary.")
        with open(paths.ALPHA_PATH) as f:
            p1wkt = f.readlines()[0]
        p1 = shapely.wkt.loads(p1wkt)

    return p1


def _return_data_locations(
    maps: list, dd: Union[pd.DataFrame, xr.Dataset], featuretype: str, logger=None
) -> Tuple[Union[float, np.array], Union[float, np.array]]:
    """Return lon, lat locations from dataset.

    Parameters
    ----------
    maps : list
        Each entry is a list of information about a dataset; the last entry is for the present source_name or dataset. Each entry contains [min_lon, max_lon, min_lat, max_lat, source_name] and possibly an additional element containing "maptype".
    dd : Union[pd.DataFrame, xr.Dataset]
        Dataset
    featuretype : str
        NCEI feature type for dataset
    logger : optional
        logger, by default None

    Returns
    -------
    tuple
        * lons: float or array of floats
        * lats: float or array of floats
    """

    min_lon, max_lon, min_lat, max_lat, source_name = maps[-1][:5]

    # logic for one or multiple lon/lat locations
    if (
        min_lon != max_lon
        or min_lat != max_lat
        or featuretype == "trajectory"
        or featuretype == "trajectoryProfile"
    ):
        if logger is not None:
            logger.info(
                f"Source {source_name} is not stationary so using multiple locations."
            )
        lons, lats = (
            dd.cf["longitude"].values,
            dd.cf["latitude"].values,
        )
    else:
        lons, lats = min_lon, max_lat

    return lons, lats


def _is_outside_boundary(
    p1: shapely.Polygon, lon: float, lat: float, source_name: str, logger=None
) -> bool:
    """Checks point to see if is outside model domain.

    This currently assumes that the dataset is fixed in space.

    Parameters
    ----------
    p1 : shapely.Polygon
        Model domain boundary
    lon : float
        Longitude of point to compare with model domain boundary
    lat : float
        Latitude of point to compare with model domain boundary
    source_name : str
        Name of dataset within cat to examine.
    logger : optional
        logger, by default None

    Returns
    -------
    bool
        True if lon, lat point is outside the model domain boundary, otherwise False.
    """

    # BUT — might want to just use nearest point so make this optional
    point = Point(lon, lat)
    if not p1.contains(point):
        msg = f"Dataset {source_name} at lon {lon}, lat {lat} not located within model domain. Skipping dataset.\n"
        if logger is not None:
            logger.warning(msg)
        return True
    else:
        return False


def _process_model(
    dsm2: xr.Dataset,
    preprocess: bool,
    need_xgcm_grid: bool,
    kwargs_xroms: dict,
    logger=None,
) -> Tuple[xr.Dataset, Grid, bool]:
    """Process model output a second time, possibly.

    Parameters
    ----------
    dsm2 : xr.Dataset
        Model output Dataset, already narrowed in time.
    preprocess : bool
        True to preprocess.
    need_xgcm_grid : bool
        True if need to find `xgcm` grid object.
    kwargs_xroms : dict
        Keyword arguments to pass to xroms.
    logger : optional
        logger, by default None

    Returns
    -------
    tuple
        * dsm2: Model output, possibly modified
        * grid: xgcm grid object or None
        * preprocessed: bool that is True if model output was processed in this function
    """
    preprocessed = False

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

                if logger is not None:
                    logger.info(
                        "setting up for model output with xroms, might take a few minutes..."
                    )
                dsm2, grid = xroms.roms_dataset(dsm2, **kwargs_xroms)
                dsm2.xroms.set_grid(grid)
                check_dataset(dsm2)

        # now has been preprocessed
        preprocessed = True
    else:
        grid = None

    return dsm2, grid, preprocessed


def _return_mask(
    mask: xr.DataArray,
    dsm: xr.Dataset,
    lon_name: str,
    wetdry: bool,
    key_variable_data: str,
    paths: Paths,
    logger=None,
) -> xr.DataArray:
    """Find or calculate and check mask.

    Parameters
    ----------
    mask : xr.DataArray or None
        Values are 1 for active cells and 0 for inactive grid cells in the model dsm.
    dsm : xr.Dataset
        Model output Dataset
    lon_name : str
        variable name for longitude in dsm.
    wetdry : bool
        Adjusts the logic in the search for mask such that if True, selected mask must include "wetdry" in name and will use first time step.
    key_variable_data : str
        Key name of variable
    paths : Paths
        Paths to files and directories for this project.
    logger
        optional

    Returns
    -------
    DataArray
        Mask
    """

    # take out relevant variable and identify mask if available (otherwise None)
    # this mask has to match dam for em.select()
    if mask is None:
        if paths.MASK_PATH(key_variable_data).is_file():
            if logger is not None:
                logger.info(
                    f"Using cached mask from {paths.MASK_PATH(key_variable_data)}."
                )
            mask = xr.open_dataarray(paths.MASK_PATH(key_variable_data))
        else:
            if logger is not None:
                logger.info(
                    f"Finding and saving mask to cache to {paths.MASK_PATH(key_variable_data)}."
                )
            # # dam variable might not be in Dataset itself, but its coordinates probably are.
            # mask = get_mask(dsm, dam.name)
            mask = get_mask(dsm, lon_name, wetdry=wetdry)
            assert mask is not None
            mask.to_netcdf(paths.MASK_PATH(key_variable_data))

    # there should not be any nans in the mask!
    if mask.isnull().any():
        raise ValueError(
            f"""there are nans in your mask — better fix something.
                            The cached version is at {paths.MASK_PATH(key_variable_data)}.
                            """
        )

    return mask


def _select_process_save_model(
    select_kwargs: dict,
    source_name: str,
    model_source_name: str,
    model_file_name: pathlib.Path,
    save_horizontal_interp_weights: bool,
    key_variable_data: str,
    maps: list,
    paths: Paths,
    logger=None,
) -> Tuple[xr.Dataset, bool, list]:
    """Select model output, process, and save to file

    Parameters
    ----------
    select_kwargs : dict
        Keyword arguments to send to `em.select()` for model extraction
    source_name : str
        Name of dataset within cat to examine.
    model_source_name : str
        Source name for model in the model catalog
    model_file_name : pathlib.Path
        Path to where to save model output
    save_horizontal_interp_weights : bool
        Default True. Whether or not to save horizontal interp info like Delaunay triangulation to file. Set to False to not save which is useful for testing.
    key_variable_data : str
        Name of variable to select, to be interpreted with cf-xarray
    maps : list
        Each entry is a list of information about a dataset; the last entry is for the present source_name or dataset. Each entry contains [min_lon, max_lon, min_lat, max_lat, source_name] and possibly an additional element containing "maptype".
    paths : Paths
        Paths object for finding paths to use.
    logger : logger, optional
        Logger for messages.

    Returns
    -------
    tuple
        * model_var: xr.Dataset with selected model output
        * skip_dataset: True if we should skip this dataset due to checks in this function
        * maps: Same as input except might be missing final entry if skipping this dataset
    """

    dam = select_kwargs.pop("dam")

    skip_dataset = False

    # use pickle of triangulation from project dir if available
    tri_name = paths.PROJ_DIR / "tri.pickle"
    if (
        select_kwargs["horizontal_interp"]
        and select_kwargs["horizontal_interp_code"] == "delaunay"
        and tri_name.is_file()
    ):
        import pickle

        if logger is not None:
            logger.info(
                f"Using previously-calculated Delaunay triangulation located at {tri_name}."
            )

        with open(tri_name, "rb") as handle:
            tri = pickle.load(handle)
    else:
        tri = None

    # add tri to select_kwargs to use in em.select
    select_kwargs["triangulation"] = tri

    if logger is not None:
        logger.info(
            f"Selecting model output at locations to match dataset {source_name}."
        )

    model_var, kwargs_out = em.select(dam, **select_kwargs)

    # save pickle of triangulation to project dir
    if (
        select_kwargs["horizontal_interp"]
        and select_kwargs["horizontal_interp_code"] == "delaunay"
        and not tri_name.is_file()
        and save_horizontal_interp_weights
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
    if select_kwargs["horizontal_interp"]:
        msg += f"""
    Interpolation coordinates used for horizontal interpolation are {kwargs_out["interp_coords"]}."""
    else:
        msg += f"""
    Output information from finding nearest neighbors to requested points are {kwargs_out}."""
    if logger is not None:
        logger.info(msg)

    # Use distances from xoak to give context to how far the returned model points might be from
    # the data locations
    if not select_kwargs["horizontal_interp"]:
        distance = kwargs_out["distances"]
        if (distance > 5).any():
            if logger is not None:
                logger.warning(
                    "Distance between nearest model location and data location for source %s is over 5 km with a distance of %s",
                    source_name,
                    str(float(distance)),
                )
        elif (distance > 100).any():
            msg = f"Distance between nearest model location and data location for source {source_name} is over 100 km with a distance of {float(distance)}. Skipping dataset.\n"
            if logger is not None:
                logger.warning(msg)
            maps.pop(-1)
            skip_dataset = True

    if model_var.cf["T"].size == 0:
        # model output isn't available to match data
        # data must not be in the space/time range of model
        maps.pop(-1)
        if logger is not None:
            logger.warning(
                "Model output is not present to match dataset %s.",
                source_name,
            )
        skip_dataset = True

    # this is trying to drop z_rho type coordinates to not save an extra time series
    # do need to use "vertical" here instead of "Z" since "Z" will be s_rho and we want
    # to keep that
    if (
        select_kwargs["Z"] is not None
        and not select_kwargs["vertical_interp"]
        and "vertical" in model_var.cf.coordinates
    ):
        if logger is not None:
            logger.info("Trying to drop vertical coordinates time series")
        if model_var.cf["vertical"].ndim > 2:
            model_var = model_var.drop_vars(model_var.cf["vertical"].name)

    # try rechunking to avoid killing kernel
    if model_var.dims == (model_var.cf["T"].name,):
        # for simple case of only time, just rechunk into pieces if no chunks
        if model_var.chunks == ((model_var.size,),):
            if logger is not None:
                logger.info(f"Rechunking model output...")
            model_var = model_var.chunk({model_var.cf["T"].name: 1})

    if logger is not None:
        logger.info(f"Loading model output...")
    model_var = model_var.compute()
    # depths shouldn't need to be saved if interpolated since then will be a dimension
    if select_kwargs["Z"] is not None and not select_kwargs["vertical_interp"]:
        # find Z index
        if "Z" in dam.cf.axes:
            zkey = dam.cf["Z"].name
            iz = list(dam.cf["Z"].values).index(model_var[zkey].values)
            model_var[f"{zkey}_index"] = iz
            # if we chose an index maybe there is no vertical? experimental
            if "vertical" not in model_var.cf:
                model_var[f"{zkey}_index"].attrs["positive"] = dam.cf["vertical"].attrs[
                    "positive"
                ]
        else:
            raise KeyError("Z missing from dam axes")
    if not select_kwargs["horizontal_interp"]:
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
        if "longitude" not in model_var.cf:
            # if dam.cf["longitude"].name not in model_var.coords:
            # if model_var.ndim == 1 and len(model_var[model_var.dims[0]]) == lons.size:
            if isinstance(select_kwargs["longitude"], (float, int)):
                attrs = dict(
                    axis="X",
                    units="degrees_east",
                    standard_name="longitude",
                )
                model_var[dam.cf["longitude"].name] = select_kwargs["longitude"]
                model_var[dam.cf["longitude"].name].attrs = attrs
            elif (
                model_var.ndim == 1
                and len(model_var[model_var.dims[0]]) == select_kwargs["longitude"].size
            ):
                attrs = dict(
                    axis="X",
                    units="degrees_east",
                    standard_name="longitude",
                )
                model_var[dam.cf["longitude"].name] = (
                    model_var.dims[0],
                    select_kwargs["longitude"],
                    attrs,
                )
        if "latitude" not in model_var.cf:
            # if dam.cf["latitude"].name not in model_var.dims:
            if isinstance(select_kwargs["latitude"], (float, int)):
                model_var[dam.cf["latitude"].name] = select_kwargs["latitude"]
                attrs = dict(
                    axis="Y",
                    units="degrees_north",
                    standard_name="latitude",
                )
                model_var[dam.cf["latitude"].name].attrs = attrs
            elif (
                model_var.ndim == 1
                and len(model_var[model_var.dims[0]]) == select_kwargs["latitude"].size
            ):
                attrs = dict(
                    axis="Y",
                    units="degrees_north",
                    standard_name="latitude",
                )
                model_var[dam.cf["latitude"].name] = (
                    model_var.dims[0],
                    select_kwargs["latitude"],
                    attrs,
                )
    attrs = {
        "key_variable": key_variable_data,
        "vertical_interp": str(select_kwargs["vertical_interp"]),
        "interpolate_horizontal": str(select_kwargs["horizontal_interp"]),
        "model_source_name": model_source_name,
        "source_name": source_name,
    }
    if select_kwargs["horizontal_interp"]:
        attrs.update(
            {
                "horizontal_interp_code": select_kwargs["horizontal_interp_code"],
            }
        )
    model_var.attrs.update(attrs)

    if select_kwargs["Z"] is None:
        no_Z = True
    else:
        no_Z = False

    model_var = model_var.cf.guess_coord_axis()

    try:
        check_dataset(model_var, no_Z=no_Z)
    except KeyError:
        # see if I can fix it
        model_var = fix_dataset(model_var, dam)
        check_dataset(model_var, no_Z=no_Z)

    if logger is not None:
        logger.info(f"Saving model output to file...")
    model_var.to_netcdf(model_file_name)

    return model_var, skip_dataset, maps


def run(
    catalogs: Union[str, Catalog, Sequence],
    project_name: str,
    key_variable: Union[str, dict],
    model_name: Union[str, Catalog],
    vocabs: Optional[Union[str, Vocab, Sequence, PurePath]] = None,
    vocab_labels: Optional[Union[str, Path, dict]] = None,
    ndatasets: Optional[int] = None,
    kwargs_map: Optional[Dict] = None,
    verbose: bool = True,
    mode: str = "w",
    testing: bool = False,
    alpha: int = 5,
    dd: int = 2,
    preprocess: bool = False,
    need_xgcm_grid: bool = False,
    xcmocean_options: Optional[dict] = None,
    kwargs_xroms: Optional[dict] = None,
    locstream: bool = True,
    interpolate_horizontal: bool = True,
    horizontal_interp_code="delaunay",
    save_horizontal_interp_weights: bool = True,
    want_vertical_interp: bool = False,
    extrap: bool = False,
    model_source_name: Optional[str] = None,
    catalog_source_names=None,
    user_min_time: Optional[Union[str, pd.Timestamp]] = None,
    user_max_time: Optional[Union[str, pd.Timestamp]] = None,
    check_in_boundary: bool = True,
    tidal_filtering: Optional[Dict[str, bool]] = None,
    ts_mods: Optional[list] = None,
    model_only: bool = False,
    plot_map: bool = True,
    no_Z: bool = False,
    skip_mask: bool = False,
    wetdry: bool = False,
    plot_count_title: bool = True,
    cache_dir: Optional[Union[str, PurePath]] = None,
    return_fig: bool = False,
    override_model: bool = False,
    override_processed: bool = False,
    override_stats: bool = False,
    override_plot: bool = False,
    plot_description: Optional[str] = None,
    kwargs_plot: Optional[Dict] = None,
    skip_key_variable_check: bool = False,
    **kwargs,
):
    """Run the model-data comparison.

    Note that timezones are assumed to match between the model output and data.

    To avoid calculating a mask you need to input `skip_mask=True`, `check_in_boundary=False`, and `plot_map=False`.

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
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache. This should be supplied, however it is made optional because it could be provided by setting it outside of the OMSA code.
    vocab_labels : str, dict, Path, optional
        Ultimately a dictionary whose keys match the input vocab and values have strings to be used in plot labels, such as "Sea water temperature [C]" for the key "temp". They can be input from a stored file or as a dict.
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
    locstream: boolean, optional
        Which type of interpolation to do, passed to em.select():

        * False: 2D array of points with 1 dimension the lons and the other dimension the lats.
        * True: lons/lats as unstructured coordinate pairs (in xESMF language, LocStream).
    interpolate_horizontal : bool, optional
        If True, interpolate horizontally. Otherwise find nearest model points.
    horizontal_interp_code: str
        Default "xesmf" to use package ``xESMF`` for horizontal interpolation, which is probably better if you need to interpolate to many points. To use ``xESMF`` you have install it as an optional dependency. Input "tree" to use BallTree to find nearest 3 neighbors and interpolate using barycentric coordinates. This has been tested for interpolating to 3 locations so far. Input "delaunay" to use a delaunay triangulation to find the nearest triangle points and interpolate the same as with "tree" using barycentric coordinates. This should be faster when you have more points to interpolate to, especially if you save and reuse the triangulation.
    save_horizontal_interp_weights : bool
        Default True. Whether or not to save horizontal interp info like Delaunay triangulation to file. Set to False to not save which is useful for testing.
    want_vertical_interp: bool
        This is False unless the user wants to specify that vertical interpolation should happen. This is used in only certain cases but in those cases it is important so that it is known to interpolate instead of try to figure out a vertical level index (which is not possible currently).
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
    ts_mods : list
        list of time series modifications to apply to data and model.
    model_only: bool
        If True, reads in model output and saves to cache, then stops. Default False.
    plot_map : bool
        If False, don't plot map
    no_Z : bool
        If True, set Z=None so no vertical interpolation or selection occurs. Do this if your variable has no concept of depth, like the sea surface height.
    skip_mask : bool
        Allows user to override mask behavior and keep it as None. Good for testing. Default False. Also skips mask in p1 calculation and map plotting if set to False and those are set to True.
    wetdry : bool
        If True, insist that masked used has "wetdry" in the name and then use the first time step of that mask.
    plot_count_title : bool
        If True, have a count to match the map of the station number in the title, like "0: [station name]". Otherwise skip count.
    cache_dir: str, Path
        Pass on to omsa.paths to set cache directory location if you don't want to use the default. Good for testing.
    vocab_labels: dict, optional
        dict with keys that match input vocab for putting labels with units on the plots. User has to make sure they match both the data and model; there is no unit handling.
    return_fig: bool
        Set to True to return all outputs from this function. Use for testing. Only works if using a single source.
    override_model : bool
        Flag to force-redo model selection. Default False.
    override_processed : bool
        Flag to force-redo model and data processing. Default False.
    override_stats : bool
        Flag to force-redo stats calculation. Default False.
    override_plot : bool
        Flag to force-redo plot. If True, only redos plot itself if other files are already available. If False, only redos the plot not the other files. Default False.
    kwargs_plot : dict
        to pass to omsa plot selection and then through the omsa plot selection to the subsequent plot itself for source. If you need more fine options, run the run function per source.
    skip_key_variable_check : bool
        If True, don't check for key_variable name being in catalog source metadata.
    """

    paths = Paths(project_name, cache_dir=cache_dir)

    logger = set_up_logging(verbose, paths=paths, mode=mode, testing=testing)

    logger.info(f"Input parameters: {locals()}")

    kwargs_map = kwargs_map or {}
    kwargs_plot = kwargs_plot or {}
    kwargs_xroms = kwargs_xroms or {}
    ts_mods = ts_mods or []

    # add override_plot to kwargs_plot in case the fignames are changed later and should be checked there instead
    kwargs_plot.update({"override_plot": override_plot})

    mask = None

    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    if vocabs is not None:
        vocab = open_vocabs(vocabs, paths)
        # now we shouldn't need to worry about this for the rest of the run right?
        cfp_set_options(custom_criteria=vocab.vocab)
        cfx_set_options(custom_criteria=vocab.vocab)
    else:
        vocab = None

    # After this, we have None or a dict with key, values of vocab keys, string description for plot labels
    if vocab_labels is not None:
        vocab_labels = open_vocab_labels(vocab_labels, paths)

    # Open and check catalogs.
    cats = open_catalogs(catalogs, paths, skip_strings=["_base", "_all", "_tidecons"])

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

    # have to save this because of my poor variable naming at the moment as I make a list possible
    key_variable_orig = key_variable

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

            skip_dataset = False

            if ndatasets is None:
                msg = (
                    f"\nsource name: {source_name} ({i+1} of {ndata} for catalog {cat}."
                )
            else:
                msg = f"\nsource name: {source_name} ({i+1} of {ndatasets} for catalog {cat}."
            logger.info(msg)

            # this check doesn't work if key_data is a dict since too hard to figure out what to check then
            # change to iterable
            key_variable_list = cf_xarray.utils.always_iterable(key_variable_orig)
            if (
                "key_variables" in cat[source_name].metadata
                and all(
                    [
                        key not in cat[source_name].metadata["key_variables"]
                        for key in key_variable_list
                    ]
                )
                # and key_variable_list not in cat[source_name].metadata["key_variables"]
                # and not isinstance(key_variable_list, dict)
                and all([not isinstance(key, dict) for key in key_variable_list])
                and not skip_key_variable_check
            ):
                logger.info(
                    f"no `key_variables` key found in source metadata or at least not {key_variable}"
                )
                skip_dataset = True
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
                dsm, model_source_name = _initial_model_handling(
                    model_name, paths, model_source_name
                )
            assert isinstance(model_source_name, str)  # for mypy

            # Determine data min and max times
            user_min_time, user_max_time = pd.Timestamp(user_min_time), pd.Timestamp(
                user_max_time
            )
            model_min_time = pd.Timestamp(str(dsm.cf["T"][0].values))
            model_max_time = pd.Timestamp(str(dsm.cf["T"][-1].values))
            data_min_time, data_max_time = _find_data_time_range(cat, source_name)

            # skip this dataset if times between data and model don't align
            skip_dataset, maps = _check_time_ranges(
                source_name,
                data_min_time,
                data_max_time,
                model_min_time,
                model_max_time,
                user_min_time,
                user_max_time,
                maps,
                logger,
            )
            if skip_dataset:
                continue

            # key_variable could be a list of strings or dicts and here we loop over them if so
            obss, models, statss, key_variable_datas = [], [], [], []
            for key_variable in key_variable_list:

                # allow for possibility that key_variable is a dict with more complicated usage than just a string
                if isinstance(key_variable, dict):
                    key_variable_data = key_variable["data"]
                else:
                    key_variable_data = key_variable

                logger.info(
                    f"running {source_name} for key_variable(s) {key_variable_data} from key_variable_list {key_variable_list}\n"
                )

                # # Combine and align the two time series of variable
                # with cfp_set_options(custom_criteria=vocab.vocab):

                try:
                    dfd = cat[source_name].read()
                    if isinstance(dfd, pd.DataFrame):
                        dfd = check_dataframe(dfd, no_Z)

                except requests.exceptions.HTTPError as e:
                    logger.warning(str(e))
                    msg = f"Data cannot be loaded for dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    skip_dataset = True
                    continue

                except Exception as e:
                    logger.warning(str(e))
                    msg = f"Data cannot be loaded for dataset {source_name}. Skipping dataset.\n"
                    logger.warning(msg)
                    maps.pop(-1)
                    skip_dataset = True
                    continue

                # check for already-aligned model-data file
                fname_processed_orig = (
                    f"{cat.name}_{source_name.replace('.','_')}_{key_variable_data}"
                )
                (
                    fname_processed,
                    fname_processed_data,
                    fname_processed_model,
                    model_file_name,
                ) = _processed_file_names(
                    fname_processed_orig,
                    type(dfd),
                    user_min_time,
                    user_max_time,
                    paths,
                    ts_mods,
                    logger,
                )
                figname = (paths.OUT_DIR / f"{fname_processed.stem}").with_suffix(
                    ".png"
                )
                # in case there are multiple key_variables in key_variable_list which will be joined
                # for the figure, renamed including both names
                if len(key_variable_list) > 1:
                    figname = pathlib.Path(
                        str(figname).replace(
                            key_variable_data, "_".join(key_variable_list)
                        )
                    )

                logger.info(f"Figure name is {figname}.")

                if figname.is_file() and not override_plot:
                    logger.info(f"plot already exists so skipping dataset.")
                    continue

                # read in previously-saved processed model output and obs.
                if (
                    not override_processed
                    and fname_processed_data.is_file()
                    and fname_processed_model.is_file()
                ):

                    logger.info(
                        "Reading previously-processed model output and data for %s.",
                        source_name,
                    )
                    obs = read_processed_data_file(fname_processed_data, no_Z)
                    model = read_model_file(fname_processed_model, no_Z, dsm)
                else:

                    logger.info(
                        "No previously processed model output and data available for %s, so setting up now.",
                        source_name,
                    )

                    # take out relevant variable and identify mask if available (otherwise None)
                    # this mask has to match dam for em.select()
                    if not skip_mask:
                        mask = _return_mask(
                            mask,
                            dsm,
                            dsm.cf.coordinates["longitude"][
                                0
                            ],  # using the first longitude key is adequate
                            wetdry,
                            key_variable_data,
                            paths,
                            logger,
                        )

                    # I think these should always be true together
                    if skip_mask:
                        assert mask is None

                    # Calculate boundary of model domain to compare with data locations and for map
                    # don't need p1 if check_in_boundary False and plot_map False
                    if (check_in_boundary or plot_map) and p1 is None:
                        p1 = _return_p1(paths, dsm, mask, alpha, dd, logger)

                    # see if data location is inside alphashape-calculated polygon of model domain
                    if check_in_boundary:
                        if _is_outside_boundary(
                            p1, min_lon, min_lat, source_name, logger
                        ):
                            maps.pop(-1)
                            continue

                    # Check, prep, and possibly narrow data time range
                    dfd, maps = _check_prep_narrow_data(
                        dfd,
                        key_variable_data,
                        source_name,
                        maps,
                        vocab,
                        user_min_time,
                        user_max_time,
                        data_min_time,
                        data_max_time,
                        logger,
                    )
                    # if there were any issues in the last function, dfd should be None and we should
                    # skip this dataset
                    if dfd is None:
                        skip_dataset = True
                        continue

                    # Read in model output from cache if possible.
                    if not override_model and model_file_name.is_file():
                        logger.info("Reading model output from file.")
                        model_var = read_model_file(model_file_name, no_Z, dsm)
                        if not interpolate_horizontal:
                            distance = model_var["distance"]

                        # Is this necessary? It removes `s_rho_index` when present which causes an issue
                        # since it is "vertical" for cf
                        # model_var = model_var.cf[key_variable_data]

                        # if model_only:
                        #     logger.info("Running model only so moving on to next source...")
                        #     continue

                    # have to read in the model output
                    else:

                        # lons, lats might be one location or many
                        lons, lats = _return_data_locations(
                            maps, dfd, cat[source_name].metadata["featuretype"], logger
                        )

                        # narrow time range to limit how much model output to deal with
                        dsm2 = _narrow_model_time_range(
                            dsm,
                            user_min_time,
                            user_max_time,
                            model_min_time,
                            model_max_time,
                            data_min_time,
                            data_max_time,
                        )

                        # more processing opportunity and chance to use xroms if needed
                        dsm2, grid, preprocessed = _process_model(
                            dsm2, preprocess, need_xgcm_grid, kwargs_xroms, logger
                        )

                        # Narrow model from Dataset to DataArray here
                        # key_variable = ["xroms", "ualong", "theta"]  # and all necessary steps to get there will happen
                        # key_variable = {"accessor": "xroms", "function": "ualong", "inputs": {"theta": theta}}
                        # # HOW TO GET THETA IN THE DICT?

                        # dam might be a Dataset but it has to be on a single grid, that is, e.g., all variable on the ROMS rho grid.
                        # well, that is only partially true. em.select requires DataArrays for certain operations like vertical
                        # interpolation.
                        dam = _dam_from_dsm(
                            dsm2,
                            key_variable,
                            key_variable_data,
                            cat[source_name].metadata,
                            no_Z,
                            logger,
                        )

                        # shift if 0 to 360
                        dam = shift_longitudes(dam)  # this is fast if not needed

                        # expand 1D coordinates to 2D, so all models dealt with in OMSA are treated with 2D coords.
                        # if your model is too large to be treated with this way, subset the model first.
                        dam = coords1Dto2D(dam)  # this is fast if not needed

                        # if locstreamT then want to keep all the data times (like a CTD transect)
                        # if not, just want the unique values (like a CTD profile)
                        locstreamT = ftconfig[cat[source_name].metadata["featuretype"]][
                            "locstreamT"
                        ]
                        locstreamZ = ftconfig[cat[source_name].metadata["featuretype"]][
                            "locstreamZ"
                        ]
                        if locstreamT:
                            T = [pd.Timestamp(date) for date in dfd.cf["T"].values]
                        else:
                            T = [
                                pd.Timestamp(date)
                                for date in np.unique(dfd.cf["T"].values)
                            ]

                        # Need to have this here because if model file has previously been read in but
                        # aligned file doesn't exist yet, this needs to run to update the sign of the
                        # data depths in certain cases.
                        zkeym = dsm.cf.axes["Z"][0]
                        dfd, Z, vertical_interp = _choose_depths(
                            dfd,
                            dsm[zkeym].attrs["positive"],
                            no_Z,
                            want_vertical_interp,
                            logger,
                        )

                        select_kwargs = dict(
                            dam=dam,
                            longitude=lons,
                            latitude=lats,
                            # T=slice(user_min_time, user_max_time),
                            # T=np.unique(dfd.cf["T"].values),  # works for Datasets
                            # T=np.unique(dfd.cf["T"].values).tolist(),  # works for DataFrame
                            # T=list(np.unique(dfd.cf["T"].values)),  # might work for both
                            # T=[pd.Timestamp(date) for date in np.unique(dfd.cf["T"].values)],
                            T=T,
                            # # works for both
                            # T=None,  # changed this because wasn't working with CTD profiles. Time interpolation happens during _align.
                            Z=Z,
                            vertical_interp=vertical_interp,
                            iT=None,
                            iZ=None,
                            extrap=extrap,
                            extrap_val=None,
                            locstream=locstream,
                            locstreamT=locstreamT,
                            locstreamZ=locstreamZ,
                            # locstream_dim="z_rho",
                            weights=None,
                            mask=mask,
                            use_xoak=False,
                            horizontal_interp=interpolate_horizontal,
                            horizontal_interp_code=horizontal_interp_code,
                            xgcm_grid=grid,
                            return_info=True,
                        )
                        model_var, skip_dataset, maps = _select_process_save_model(
                            select_kwargs,
                            source_name,
                            model_source_name,
                            model_file_name,
                            save_horizontal_interp_weights,
                            key_variable_data,
                            maps,
                            paths,
                            logger,
                        )
                        if skip_dataset:
                            continue

                    if model_only:
                        logger.info("Running model only so moving on to next source...")
                        continue

                    # opportunity to modify time series data
                    # fnamemods = ""
                    from copy import deepcopy

                    ts_mods_copy = deepcopy(ts_mods)
                    # ts_mods_copy = ts_mods.copy()  # otherwise you modify ts_mods when adding data
                    for mod in ts_mods_copy:
                        logger.info(
                            f"Apply a time series modification called {mod['function']}."
                        )
                        if isinstance(dfd, pd.DataFrame):
                            dfd.set_index(dfd.cf["T"], inplace=True)

                        # this is how you include the dataset in the inputs
                        if (
                            "include_data" in mod["inputs"]
                            and mod["inputs"]["include_data"]
                        ):
                            mod["inputs"].update({"dd": dfd})
                            mod["inputs"].pop("include_data")

                        # apply ts_mod to full dataset instead of just one variable since might want
                        # to use more than one of the variables
                        # also need to overwrite Dataset since the shape of the variables might change here
                        dfd = mod["function"](dfd, **mod["inputs"])
                        # dfd[dfd.cf[key_variable_data].name] = mod["function"](
                        #     dfd.cf[key_variable_data], **mod["inputs"]
                        # )
                        if isinstance(dfd, pd.DataFrame):
                            if dfd.cf["T"].name in dfd.columns:
                                drop = True
                            else:
                                drop = False

                            dfd = dfd.reset_index(drop=drop)

                        model_var = mod["function"](model_var, **mod["inputs"])

                    # check model output for nans
                    ind_keep = np.arange(0, model_var.cf["T"].size)[
                        model_var.cf["T"].notnull()
                    ]
                    if model_var.cf["T"].name in model_var.dims:
                        model_var = model_var.isel({model_var.cf["T"].name: ind_keep})

                    # there could be a small mismatch in the length of time if times were pulled
                    # out separately
                    if np.unique(model_var.cf["T"]).size != np.unique(dfd.cf["T"]).size:
                        logger.info("Changing the timing of the model or data.")
                        # if model_var.cf["T"].size != np.unique(dfd.cf["T"]).size:
                        # if (isinstance(dfd, pd.DataFrame) and model_var.cf["T"].size != dfd.cf["T"].unique().size) or (isinstance(dfd, xr.Dataset) and model_var.cf["T"].size != dfd.cf["T"].drop_duplicates(dim=dfd.cf["T"].name).size):
                        # if len(model_var.cf["T"]) != len(dfd.cf["T"]):  # timeSeries
                        stime = pd.Timestamp(
                            max(dfd.cf["T"].values[0], model_var.cf["T"].values[0])
                        )
                        etime = pd.Timestamp(
                            min(dfd.cf["T"].values[-1], model_var.cf["T"].values[-1])
                        )
                        if stime != etime:
                            model_var = model_var.cf.sel({"T": slice(stime, etime)})

                        if isinstance(dfd, pd.DataFrame):
                            dfd = dfd.set_index(dfd.cf["T"].name)
                            dfd = dfd.loc[stime:etime]

                            # interpolate data to model times
                            # Times between data and model should already match from em.select
                            # except in the case that model output was cached in convenient time series
                            # in which case the times aren't already matched. For this case, the data
                            # also might be missing the occasional data points, and want
                            # the data index to match the model index since the data resolution might be very high.
                            # get combined index of model and obs to first interpolate then reindex obs to model
                            # otherwise only nan's come through
                            # accounting for known issue for interpolation after sampling if indices changes
                            # https://github.com/pandas-dev/pandas/issues/14297
                            # this won't run for single ctd profiles
                            if len(dfd.cf["T"].unique()) > 1:
                                model_index = model_var.cf["T"].to_pandas().index
                                model_index.name = dfd.index.name
                                ind = model_index.union(dfd.index)
                                dfd = (
                                    dfd.reindex(ind)
                                    .interpolate(method="time", limit=3)
                                    .reindex(model_index)
                                )
                                dfd = dfd.reset_index()

                        elif isinstance(dfd, xr.Dataset):
                            # interpolate data to model times
                            # model_index = model_var.cf["T"].to_pandas().index
                            # ind = model_index.union(dfd.cf["T"].to_pandas().index)
                            dfd = dfd.interp(
                                {dfd.cf["T"].name: model_var.cf["T"].values}
                            )
                            # dfd = dfd.cf.sel({"T": slice(stime, etime)})

                    # change names of model to match data so that stats will calculate without adding variables
                    # not necessary if dfd is DataFrame (i think)
                    if isinstance(dfd, (xr.Dataset, xr.DataArray)):
                        rename = {}
                        for model_dim in model_var.squeeze().dims:
                            matching_dim = [
                                data_dim
                                for data_dim in dfd.dims
                                if dfd[data_dim].size == model_var[model_dim].size
                            ][0]
                            rename.update({model_dim: matching_dim})
                        # rename = {model_var.cf[key].name: dfd.cf[key].name for key in ["T","Z","latitude","longitude"]}
                        model_var = model_var.rename(rename)

                    # Save processed data and model files
                    save_processed_files(
                        dfd, fname_processed_data, model_var, fname_processed_model
                    )
                    obs = read_processed_data_file(fname_processed_data, no_Z)
                    model = read_model_file(fname_processed_model, no_Z, dsm)

                logger.info(f"model file name is {model_file_name}.")
                if not override_model and model_file_name.is_file():
                    logger.info("Reading model output from file.")
                    model = read_model_file(fname_processed_model, no_Z, dsm)
                    if not interpolate_horizontal:
                        distance = model["distance"]
                else:
                    raise ValueError(
                        "If the processed files are available need this one too."
                    )

                if model_only:
                    logger.info("Running model only so moving on to next source...")
                    continue

                stats_fname = (paths.OUT_DIR / f"{fname_processed.stem}").with_suffix(
                    ".yaml"
                )

                if not override_stats and stats_fname.is_file():
                    logger.info("Reading from previously-saved stats file.")
                    with open(stats_fname, "r") as stream:
                        stats = yaml.safe_load(stream)

                else:
                    logger.info(f"Calculating stats for {key_variable_data}.")
                    stats = compute_stats(
                        obs.cf[key_variable_data], model.cf[key_variable_data].squeeze()
                    )
                    # stats = obs.omsa.compute_stats

                    # add distance in
                    if not interpolate_horizontal:
                        stats["dist"] = float(distance)

                    # save stats
                    save_stats(
                        source_name,
                        stats,
                        key_variable_data,
                        paths,
                        filename=stats_fname,
                    )
                    logger.info("Saved stats file.")

                # Combine across key_variable in case there was a list of inputs
                obss.append(obs)
                models.append(model)
                statss.append(stats)
                key_variable_datas.append(key_variable_data)

            # combine list of outputs in the case there is more than one key variable
            if len(obss) > 1:
                # if both key variables are in the dataset both times just take one
                # or could check to see if both key variables are in the first dataset
                if obss[0].equals(obss[1]):
                    obs = obss[0]
                else:
                    raise NotImplementedError

                # assume one key variable in each model output
                if all(
                    [
                        len(cf_xarray.accessor._get_all(model, key)) > 0
                        for model, key in zip(models, key_variable_list)
                    ]
                ):
                    # if len(cf_xarray.accessor._get_all(models[0], key_variable_list[0])) > 0 and :
                    model = xr.merge(models)
                else:
                    raise NotImplementedError

                # leave stats as a list
                stats = statss

            # if there was always just one key variable for this run, do nothing since the variables are
            # already available correctly named
            else:
                pass

            # # currently title is being set in plot.selection
            # if plot_count_title:
            #     title = f"{count}: {source_name}"
            # else:
            #     title = f"{source_name}"
            if not skip_dataset and (not figname.is_file() or override_plot):
                fig = plot.selection(
                    obs,
                    model,
                    cat[source_name].metadata["featuretype"],
                    key_variable_datas,
                    source_name,
                    stats,
                    figname,
                    plot_description,
                    vocab_labels,
                    xcmocean_options=xcmocean_options,
                    **kwargs_plot,
                )
                msg = f"Made plot for {source_name}\n."
                logger.info(msg)

            count += 1

    # map of model domain with data locations
    if plot_map:
        if len(maps) > 0:
            try:
                figname = paths.OUT_DIR / "map.png"
                plot.map.plot_map(np.asarray(maps), figname, p=p1, **kwargs_map)
            except ModuleNotFoundError:
                pass
        else:
            logger.warning("Not plotting map since no datasets to plot.")
    logger.info(
        "Finished analysis. Find plots, stats summaries, and log in %s.",
        str(paths.PROJ_DIR),
    )

    # just have option for returning info for testing and if dealing with
    # a single source
    if len(maps) == 1 and return_fig:
        # model output, processed data, processed model, stats, fig
        return fig
    # else:
    #     plt.close(fig)
