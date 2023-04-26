"""
Main run functions.
"""

import logging
import mimetypes
import warnings

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any, Dict, List, Optional, Union

import extract_model.accessor
import intake
import requests

from cf_pandas import Vocab, astype
from cf_pandas import set_options as cfp_set_options
from cf_xarray import set_options as cfx_set_options
from extract_model import preprocess
from extract_model.utils import guess_model_type
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
from numpy import asarray, sum
from pandas import DataFrame, to_datetime
from shapely.geometry import Point

from ocean_model_skill_assessor.plot import map

from .paths import CAT_PATH, PROJ_DIR, VOCAB_PATH
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
        if (
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
                logging.warning(
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

    set_up_logging(project_name, verbose, mode=mode, testing=testing)

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
        logging.info(
            f"Catalog saved to {CAT_PATH(catalog_name, project_name)} with {len(list(cat))} entries."
        )

    logging.shutdown()

    if return_cat:
        return cat


def run(
    catalogs: Union[str, Catalog, Sequence],
    project_name: str,
    key_variable: str,
    model_name: Union[str, Catalog],
    vocabs: Union[str, Vocab, Sequence, PurePath],
    ndatasets: Optional[int] = None,
    kwargs_map: Optional[Dict] = None,
    verbose: bool = True,
    mode: str = "w",
    testing: bool = False,
    alpha: int = 5,
    dd: int = 2,
):
    """Run the model-data comparison.

    Note that timezones are assumed to match between the model output and data.

    Parameters
    ----------
    catalogs : str, list, Catalog
        Catalog name(s) or list of names, or catalog object or list of catalog objects. Datasets will be accessed from catalog entries.
    project_name : str
        Subdirectory in cache dir to store files associated together.
    key_variable : str
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
    """

    set_up_logging(project_name, verbose, mode=mode, testing=testing)

    kwargs_map = kwargs_map or {}

    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    vocab = open_vocabs(vocabs)

    # Open catalogs.
    cats = open_catalogs(catalogs, project_name)

    # Warning about number of datasets
    ndata = sum([len(list(cat)) for cat in cats])
    if ndatasets is not None:
        logging.info(
            f"Note that we are using {ndatasets} datasets of {ndata} datasets. This might take awhile."
        )
    else:
        logging.info(
            f"Note that there are {ndata} datasets to use. This might take awhile."
        )

    # read in model output
    model_cat = open_catalogs(model_name, project_name)[0]
    dsm = model_cat[list(model_cat)[0]].to_dask()

    # process model output without using open_mfdataset
    # vertical coords have been an issue for ROMS and POM, related to dask and OFS models
    if guess_model_type(dsm) in ["ROMS", "POM"]:
        kwargs_pp = {"interp_vertical": False}
    else:
        kwargs_pp = {}
    dsm = preprocess(dsm, kwargs=kwargs_pp)

    with cfx_set_options(custom_criteria=vocab.vocab):
        dam = dsm.cf[key_variable]

    # take out relevant variable and identify mask if available (otherwise None)
    mask = get_mask(dsm, dam.name)

    # shift if 0 to 360
    dam = shift_longitudes(dam)

    # expand 1D coordinates to 2D, so all models dealt with in OMSA are treated with 2D coords.
    # if your model is too large to be treated with this way, subset the model first.
    dam = coords1Dto2D(dam)

    # Calculate boundary of model domain to compare with data locations and for map
    _, _, _, p1 = find_bbox(dam, mask, alpha=alpha, dd=dd)

    # loop over catalogs and sources to pull out lon/lat locations for plot
    maps = []
    count = 0  # track datasets since count is used to match on map
    for cat in cats:
        logging.info(f"Catalog {cat}.")
        for i, source_name in enumerate(list(cat)[:ndatasets]):

            if ndatasets is None:
                msg = (
                    f"\nsource name: {source_name} ({i+1} of {ndata} for catalog {cat}."
                )
            else:
                msg = f"\nsource name: {source_name} ({i+1} of {ndatasets} for catalog {cat}."
            logging.info(msg)

            min_lon = cat[source_name].metadata["minLongitude"]
            max_lon = cat[source_name].metadata["maxLongitude"]
            min_lat = cat[source_name].metadata["minLatitude"]
            max_lat = cat[source_name].metadata["maxLatitude"]

            # see if data location is inside alphashape-calculated polygon of model domain
            # This currently assumes that the dataset is fixed in space.
            point = Point(min_lon, min_lat)
            if not p1.contains(point):
                msg = f"Dataset {source_name} at lon {min_lon}, lat {min_lat} not located within model domain. Skipping dataset.\n"
                logging.warning(msg)
                continue

            maps.append([min_lon, max_lon, min_lat, max_lat, source_name])

            # if min_lon != max_lon or min_lat != max_lat:
            #     warnings.warn(
            #         f"Source {source_name} in catalog {cat.name} is not stationary so not plotting."
            #     )
            #     continue

            # take time constraints as min/max if available
            if "constraints" in cat[source_name].describe()["args"]:
                min_time = cat[source_name].describe()["args"]["constraints"]["time>="]
                max_time = cat[source_name].describe()["args"]["constraints"]["time<="]
            # use kwargs_search min/max times if available
            elif (
                "kwargs_search" in cat.metadata
                and "min_time" in cat.metadata["kwargs_search"]
            ):
                min_time = cat.metadata["kwargs_search"]["min_time"]
                max_time = cat.metadata["kwargs_search"]["max_time"]
            else:
                min_time = cat[source_name].metadata["minTime"]
                max_time = cat[source_name].metadata["maxTime"]

            # Combine and align the two time series of variable
            with cfp_set_options(custom_criteria=vocab.vocab):
                try:
                    dfd = cat[source_name].read()
                except requests.exceptions.HTTPError as e:
                    logging.warning(str(e))
                    msg = f"Data cannot be loaded for dataset {source_name}. Skipping dataset.\n"
                    logging.warning(msg)
                    maps.pop(-1)
                    continue

                if key_variable not in dfd.cf:
                    msg = f"Key variable {key_variable} cannot be identified in dataset {source_name}. Skipping dataset.\n"
                    logging.warning(msg)
                    maps.pop(-1)
                    continue

                # see if more than one column of data is being identified as key_variable
                # if more than one, log warning and then choose first
                if isinstance(dfd.cf[key_variable], DataFrame):
                    msg = f"More than one variable ({dfd.cf[key_variable].columns}) have been matched to input variable {key_variable}. The first {dfd.cf[key_variable].columns[0]} is being selected. To change this, modify the vocabulary so that the two variables are not both matched, or change the input data catalog."
                    logging.warning(msg)
                    # remove other data columns
                    for col in dfd.cf[key_variable].columns[1:]:
                        dfd.drop(col, axis=1, inplace=True)

                dfd.cf["T"] = to_datetime(dfd.cf["T"])
                dfd.set_index(dfd.cf["T"], inplace=True)
                if dfd.index.tz is not None:
                    logging.warning(
                        "Dataset %s had a timezone %s which is being removed. Make sure the timezone matches the model output.",
                        source_name,
                        str(dfd.index.tz),
                    )
                    dfd.index = dfd.index.tz_convert(None)
                    dfd.cf["T"] = dfd.index

                # make sure index is sorted ascending so time goes forward
                dfd = dfd.sort_index()

                # check if all of variable is nan
                if dfd.cf[key_variable].isnull().all():
                    msg = f"All values of key variable {key_variable} are nan in dataset {source_name}. Skipping dataset.\n"
                    logging.warning(msg)
                    maps.pop(-1)
                    continue

            # Pull out nearest model output to data
            kwargs = dict(
                longitude=min_lon,
                latitude=min_lat,
                T=slice(min_time, max_time),
                Z=0,
                method="nearest",
            )

            # xoak doesn't work for 1D lon/lat coords
            if dam.cf["longitude"].ndim == dam.cf["latitude"].ndim == 1:
                # This shouldn't happen anymore, so make note if it does
                msg = "1D coordinates were found for this model but that should not be possible anymore."
                raise ValueError(msg)
                # if isinstance(kwargs["T"], slice):
                #     Targ = kwargs.pop("T")
                #     model_var = dam.cf.sel(T=Targ)
                # else:
                #     model_var = dam

                # # find indices representing mask
                # import numpy as np
                # import xarray as xr
                # eta, xi = np.where(mask.values)
                # # make advanced indexer to flatten arrays
                # model_var = model_var.cf.isel(
                #     X=xr.DataArray(xi, dims="loc"), Y=xr.DataArray(eta, dims="loc")
                # )

                # model_var = model_var.cf.sel(**kwargs)
                # # calculate distance
                # # calculate distance for the 1 point: model location vs. requested location
                # # https://stackoverflow.com/questions/56862277/interpreting-sklearn-haversine-outputs-to-kilometers
                # earth_radius = 6371  # km
                # pts = deg2rad([[kwargs["latitude"], kwargs["longitude"]], [model_var.cf["latitude"], model_var.cf["longitude"]]])
                # tree = BallTree(pts, metric = 'haversine')
                # ind, results = tree.query_radius(pts, r=earth_radius, return_distance=True)
                # distance = results[0][1]*earth_radius  # distance between points in km
            elif dam.cf["longitude"].ndim == dam.cf["latitude"].ndim == 2:
                # time slices can't be used with `method="nearest"`, so separate out
                # add "distances_name" to send to `sel2dcf`
                # also send mask in so it can be accounted for in finding nearest model point
                if isinstance(kwargs["T"], slice):
                    Targ = kwargs.pop("T")
                    model_var = dam.cf.sel(T=Targ)
                else:
                    model_var = dam

                model_var = model_var.em.sel2dcf(
                    mask=mask, distances_name="distance", **kwargs
                )  # .to_dataset()

                # downsize to DataArray
                distance = model_var["distance"]
                model_var = model_var[dam.name]

            # Use distances from xoak to give context to how far the returned model points might be from
            # the data locations
            if distance > 5:
                logging.warning(
                    "Distance between nearest model location and data location for source %s is over 5 km with a distance of %s",
                    source_name,
                    str(distance.values),
                )
            elif distance > 100:
                msg = f"Distance between nearest model location and data location for source {source_name} is over 100 km with a distance of {distance.values}. Skipping dataset.\n"
                logging.warning(msg)
                maps.pop(-1)
                continue

            if len(model_var.cf["T"]) == 0:
                # model output isn't available to match data
                # data must not be in the space/time range of model
                maps.pop(-1)
                logging.warning(
                    "Model output is not present to match dataset %s.",
                    source_name,
                )
                continue

            # Combine and align the two time series of variable
            with cfp_set_options(custom_criteria=vocab.vocab):
                df = _align(dfd.cf[key_variable], model_var)
                y_name = model_var.name

            # pull out depth at surface?

            # Where to save stats to?
            stats = df.omsa.compute_stats

            # add distance in
            stats["dist"] = float(distance)
            save_stats(source_name, stats, project_name, key_variable)

            # Write stats on plot
            figname = PROJ_DIR(project_name) / f"{source_name}_{key_variable}.png"
            df.omsa.plot(
                title=f"{count}: {source_name}",
                ylabel=y_name,
                figname=figname,
                stats=stats,
            )
            msg = f"Plotted time series for {source_name}\n."
            logging.info(msg)

            count += 1

    # map of model domain with data locations
    if len(maps) > 0:
        try:
            figname = PROJ_DIR(project_name) / "map.png"
            map.plot_map(asarray(maps), figname, p=p1, **kwargs_map)
        except ModuleNotFoundError:
            pass
    else:
        logging.warning("Not plotting map since no datasets to plot.")
    logging.info(
        "Finished analysis. Find plots, stats summaries, and log in %s.",
        str(PROJ_DIR(project_name)),
    )
    logging.shutdown()
