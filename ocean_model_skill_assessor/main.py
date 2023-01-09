"""
Main run functions.
"""

import mimetypes
import warnings

from collections.abc import Sequence
from pathlib import PurePath
from typing import Any, DefaultDict, Dict, List, Optional, Union

import cf_pandas as cfp
import cf_xarray as cfx
import extract_model as em
import intake
import numpy as np
import pandas as pd
import xarray as xr

from cf_pandas import Vocab
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
from tqdm import tqdm

import ocean_model_skill_assessor as omsa

from ocean_model_skill_assessor.plot import map, time_series

from .utils import kwargs_search_from_model


try:
    import cartopy

    CARTOPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CARTOPY_AVAILABLE = False  # pragma: no cover


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

    Pass keywords for xarray for model output into the catalog through kwargs_xarray.

    kwargs_open and metadata must be the same for all filenames. If it is not, make multiple catalogs and you can input them individually into the run command.

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
        Keyword arguments to pass on to local catalof for model for xr.open_mfdataset call or pandas open_csv.
    skip_entry_metadata : bool, optional
        This is useful for testing in which case we don't want to actually read the file. If inputting kwargs_xarray, you may want to set this to True since you are presumably making a catalog file for a model.

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
        elif "thredds" in filename and "dodsC" in filename:
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
            dd.cf["T"] = pd.to_datetime(dd.cf["T"])
            dd.set_index(dd.cf["T"], inplace=True)
            if dd.index.tz is not None:
                warnings.warn(
                    f"Dataset {source} had a timezone {dd.index.tz} which is being removed. Make sure the timezone matches the model output.",
                    RuntimeWarning,
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
    vocab: Optional[Union[cfp.Vocab, str, PurePath]] = None,
    return_cat: bool = True,
    save_cat: bool = False,
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
        Keyword arguments to input to search on the server before making the catalog. These are not used with `make_local_catalog()`; only for catalog types erddap and axds.
        Options are:
        * to search by bounding box: include all of min_lon, max_lon, min_lat, max_lat: (int, float). Longitudes must be between -180 to +180.
        * to search within a datetime range: include both of min_time, max_time: interpretable datetime string, e.g., "2021-1-1"
        * to search using a textual keyword: include `search_for` as a string.
        * model_name can be input in place of either the spatial box or the time range or both in which case those values will be found from the model output.
    kwargs_open : dict, optional
        Keyword arguments to save into local catalog for model to pass on to xr.open_mfdataset call or pandas open_csv. Only for use with catalog_type=local.
    vocab : dict, optional
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for.
    return_cat : bool, optional
        Return catalog. For when using as a Python package instead of with command line.
    save_cat: bool, optional
        Save catalog to disk into project directory under catalog_name.
    """

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

    # Should I require vocab if nickname is not None?
    # if vocab is None:
    #     # READ IN DEFAULT AND SET VOCAB
    #     vocab = cfp.Vocab("vocabs/general")

    # elif isinstance(vocab, str):
    #     vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))

    if isinstance(vocab, str):
        vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))
    elif isinstance(vocab, PurePath):
        vocab = cfp.Vocab(vocab)

    if description is None:
        description = f"Catalog of type {catalog_type}."

    if catalog_type == "local":
        catalog_name = "local_cat" if catalog_name is None else catalog_name
        if "filenames" not in kwargs:
            raise ValueError("For `catalog_type=='local'`, must input `filenames`.")
        filenames = kwargs["filenames"]
        kwargs.pop("filenames")
        cat = make_local_catalog(
            cfp.astype(filenames, list),
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
            with cfp.set_options(custom_criteria=vocab.vocab):
                cat = intake.open_erddap_cat(
                    kwargs_search=kwargs_search,
                    name=catalog_name,
                    description=description,
                    metadata=metadata,
                    **kwargs,
                )
        else:
            # import pdb; pdb.set_trace()
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
            with cfp.set_options(custom_criteria=vocab.vocab):
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
        cat.save(omsa.CAT_PATH(catalog_name, project_name))
        print(
            f"Catalog saved to {omsa.CAT_PATH(catalog_name, project_name)} with {len(list(cat))} entries."
        )

    if return_cat:
        return cat


def run(
    catalogs: Union[str, Catalog, Sequence],
    project_name: str,
    key_variable: str,
    model_name: str,
    vocabs: Union[str, Vocab, Sequence],
    ndatasets: Optional[int] = None,
    kwargs_map: Optional[Dict] = None,
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
    model_name : str, Path
        Name of catalog for model output, created with `make_catalog` call.
    vocabs : str, list, Vocab, optional
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
    ndatasets : int, optional
        Max number of datasets from each input catalog to use.
    kwargs_map : dict, optional
        Keyword arguments to pass on to omsa.plot.map.plot_map call.
    """

    kwargs_map = kwargs_map or {}

    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    vocabs = cfp.always_iterable(vocabs)
    if isinstance(vocabs[0], str):
        vocab = cfp.merge([Vocab(omsa.VOCAB_PATH(v)) for v in vocabs])
    elif isinstance(vocabs[0], Vocab):
        vocab = cfp.merge(vocabs)
    else:
        raise ValueError(
            "Vocab(s) should be input as string paths or Vocab objects or Sequence thereof."
        )

    # Open catalogs.
    catalogs = cfp.always_iterable(catalogs)
    if isinstance(catalogs[0], str):
        cats = [
            intake.open_catalog(omsa.CAT_PATH(catalog_name, project_name))
            for catalog_name in cfp.astype(catalogs, list)
        ]
    elif isinstance(catalogs[0], Catalog):
        cats = catalogs
    else:
        raise ValueError(
            "Catalog(s) should be input as string paths or Catalog objects or Sequence thereof."
        )

    # Warning about number of datasets
    ndata = np.sum([len(list(cat)) for cat in cats])
    if ndatasets is not None:
        print(
            f"Note that we are using {ndatasets} datasets of {ndata} datasets. This might take awhile."
        )
    else:
        print(f"Note that there are {ndata} datasets to use. This might take awhile.")

    # read in model output
    model_cat = intake.open_catalog(omsa.CAT_PATH(model_name, project_name))
    dsm = model_cat[list(model_cat)[0]].to_dask()

    # use only one variable from model
    with cfx.set_options(custom_criteria=vocab.vocab):
        dam = dsm.cf[key_variable]

    # shift if 0 to 360
    if dam.cf["longitude"].max() > 180:
        lkey = dam.cf["longitude"].name
        dam = dam.assign_coords(lon=(((dam[lkey] + 180) % 360) - 180))
        # rotate arrays so that the locations and values are -180 to 180
        # instead of 0 to 180 to -180 to 0
        dam = dam.roll(lon=int((dam[lkey] < 0).sum()), roll_coords=True)
        print(
            "Longitudes are being shifted because they look like they are not -180 to 180."
        )

    # loop over catalogs and sources to pull out lon/lat locations for plot
    maps = []
    count = 0  # track datasets since count is used to match on map
    for cat in tqdm(cats):
        print(f"Catalog {cat}.")
        # for source_name in tqdm(list(cat)[-ndatasets:]):
        for source_name in tqdm(list(cat)[:ndatasets]):

            min_lon = cat[source_name].metadata["minLongitude"]
            max_lon = cat[source_name].metadata["maxLongitude"]
            min_lat = cat[source_name].metadata["minLatitude"]
            max_lat = cat[source_name].metadata["maxLatitude"]

            maps.append([min_lon, max_lon, min_lat, max_lat, source_name])

            # if min_lon != max_lon or min_lat != max_lat:
            #     # import pdb; pdb.set_trace()
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
            with cfp.set_options(custom_criteria=vocab.vocab):
                print("source name: ", source_name)
                dfd = cat[source_name].read()
                if key_variable not in dfd.cf:
                    warnings.warn(
                        f"Key variable {key_variable} cannot be identified in dataset {source_name}. Skipping dataset.",
                        RuntimeWarning,
                    )
                    maps.pop(-1)
                    continue

                dfd.cf["T"] = pd.to_datetime(dfd.cf["T"])
                dfd.set_index(dfd.cf["T"], inplace=True)
                if dfd.index.tz is not None:
                    warnings.warn(
                        f"Dataset {source_name} had a timezone {dfd.index.tz} which is being removed. Make sure the timezone matches the model output.",
                        RuntimeWarning,
                    )
                    dfd.index = dfd.index.tz_convert(None)
                    dfd.cf["T"] = dfd.index

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
                # time slices can't be used with `method="nearest"`, so separate out
                if isinstance(kwargs["T"], slice):
                    Targ = kwargs.pop("T")
                    model_var = dam.cf.sel(**kwargs)  # .to_dataset()
                    model_var = model_var.cf.sel(T=Targ)
                else:
                    model_var = dam.cf.sel(**kwargs)  # .to_dataset()

            elif dam.cf["longitude"].ndim == dam.cf["latitude"].ndim == 2:
                # time slices can't be used with `method="nearest"`, so separate out
                if isinstance(kwargs["T"], slice):
                    Targ = kwargs.pop("T")
                    model_var = dam.em.sel2dcf(**kwargs)  # .to_dataset()
                    model_var = model_var.cf.sel(T=Targ)
                else:
                    model_var = dam.em.sel2dcf(**kwargs)  # .to_dataset()

            if model_var.size == 0:
                # model output isn't available to match data
                # data must not be in the space/time range of model
                maps.pop(-1)
                warnings.warn(
                    f"Model output is not present to match dataset {source_name}.",
                    RuntimeWarning,
                )
                continue

            # Combine and align the two time series of variable
            with cfp.set_options(custom_criteria=vocab.vocab):
                df = omsa.stats._align(dfd.cf[key_variable], model_var)

            # pull out depth at surface?

            # Where to save stats to?
            stats = df.omsa.compute_stats
            omsa.stats.save_stats(source_name, stats, project_name)

            # Write stats on plot
            figname = omsa.PROJ_DIR(project_name) / f"{source_name}_{key_variable}.png"
            df.omsa.plot(
                title=f"{count}: {source_name}",
                ylabel=dam.name,
                figname=figname,
                stats=stats,
            )

            count += 1

    # map of model domain with data locations
    if CARTOPY_AVAILABLE and len(maps) > 0:
        figname = omsa.PROJ_DIR(project_name) / "map.png"
        omsa.plot.map.plot_map(np.asarray(maps), figname, dsm, **kwargs_map)
    else:
        print(
            "Not plotting map since cartopy is not installed or no datasets to work with."
        )
    print(f"Finished analysis. Find plots in {omsa.PROJ_DIR(project_name)}.")
