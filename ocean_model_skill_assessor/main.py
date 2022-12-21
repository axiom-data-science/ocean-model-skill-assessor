"""
Main run functions.
"""

import mimetypes
import pathlib
import warnings

from collections.abc import Sequence
from typing import Any, DefaultDict, Dict, List, Optional, Union

import cf_pandas as cfp
import cf_xarray
import extract_model as em
import intake
import numpy as np
import pandas as pd
import xarray as xr

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
    skip_entry_metadata: bool = False,
) -> Catalog:
    """Make an intake catalog from specified data files.

    Parameters
    ----------
    filenames : list of paths
        Where to find dataset(s) from which to make local catalog.
    name : str, optional
        Name for catalog.
    description : str, optional
        Description for catalog.
    metadata : dict, optional
        Metadata for catalog.
    skip_entry_metadata : bool, optional
        This is useful for testing in which case we don't want to actually read the file.

    Returns
    -------
    Catalog
        Intake catalog with an entry for each dataset represented by a filename.
    """

    sources = []
    for filename in filenames:
        mtype = mimetypes.guess_type(filename)[0]
        if (mtype is not None and "csv" in mtype) or ".csv" in filename:
            source = getattr(intake, "open_csv")(filename)
        elif (mtype is not None and "netcdf" in mtype) or ".netcdf" in filename:
            source = getattr(intake, "open_netcdf")(filename)
        if not skip_entry_metadata:
            dd = source.read()
            # set up some basic metadata for each source
            source.metadata = {
                "minLongitude": float(dd.cf["longitude"].min()),
                "minLatitude": float(dd.cf["latitude"].min()),
                "maxLongitude": float(dd.cf["longitude"].max()),
                "maxLatitude": float(dd.cf["latitude"].max()),
                "minTime": str(dd.cf["T"].min()),
                "maxTime": str(dd.cf["T"].max()),
            }
        sources.append(source)

    # create dictionary of catalog entries
    entries = {
        pathlib.PurePath(source._urlpath).stem: LocalCatalogEntry(
            name=pathlib.PurePath(source._urlpath).stem,
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
        metadata=metadata,
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
    vocab: Optional[Union[cfp.Vocab, str, pathlib.PurePath]] = None,
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
        Keyword arguments to input to search on the server before making the catalog. These are not used with `make_local_catalog()`.
        Options are:
        * to search by bounding box: include all of min_lon, max_lon, min_lat, max_lat: (int, float). Longitudes must be between -180 to +180.
        * to search within a datetime range: include both of min_time, max_time: interpretable datetime string, e.g., "2021-1-1"
        * to search using a textual keyword: include `search_for` as a string.
        * model_path can be input in place of either the spatial box or the time range or both in which case those values will be found from the model output.
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

    kwargs = {} if kwargs is None else kwargs
    kwargs_search = {} if kwargs_search is None else kwargs_search

    # get spatial and/or temporal search terms from model if desired
    kwargs_search = kwargs_search_from_model(kwargs_search)

    # Should I require vocab if nickname is not None?
    # if vocab is None:
    #     # READ IN DEFAULT AND SET VOCAB
    #     vocab = cfp.Vocab("vocabs/general")

    # elif isinstance(vocab, str):
    #     vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))

    if isinstance(vocab, str):
        vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))
    elif isinstance(vocab, pathlib.PurePath):
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
        print(f"Catalog saved to {omsa.CAT_PATH(catalog_name, project_name)}.")

    if return_cat:
        return cat


def run(
    catalog_names: Union[Sequence, str, pathlib.PurePath],
    project_name: str,
    key_variable: str,
    model_path: str,
    vocabs: Union[str, list, cfp.Vocab],
    ndatasets: int = -1,
):
    """Run the model-data comparison.

    Note that timezones are assumed to match between the model output and data.

    Parameters
    ----------
    catalog_names : str, Path, list
        Catalog name(s) or path(s). Datasets will be accessed from catalog entries.
    project_name : str
        Subdirectory in cache dir to store files associated together.
    key_variable : str
        Key in vocab(s) representing variable to compare between model and datasets.
    model_path : str, Path
        Where to find model output. Must be readable by xarray.open_mfdataset() (will be converted to list if needed).
    vocabs : str, list, optional
        Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
    ndatasets : int, optional
        Max number of datasets from each input catalog to use.
    """

    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    if isinstance(vocabs, str):
        vocab = cfp.Vocab(omsa.VOCAB_PATH(vocabs))
    elif isinstance(vocabs, Sequence):
        if isinstance(vocabs[0], str):
            # vocabs = []
            # for v in vocabs:
            #     vocabs.append(cfp.Vocab(omsa.VOCAB_PATH(v)))
            vocab = cfp.merge([cfp.Vocab(omsa.VOCAB_PATH(v)) for v in vocabs])
        elif isinstance(vocab[0], cfp.Vocab):
            vocab = cfp.merge(vocabs)

    # read in model output
    dsm = xr.open_mfdataset(cfp.astype(model_path, list), preprocess=em.preprocess)

    # use only one variable from model
    dam = dsm.cf[key_variable]

    # shift if 0 to 360
    if dam.cf["longitude"].max() > 180:
        lkey = dam.cf["longitude"].name
        dam[lkey] = dam[lkey] - 360

    # Open catalogs.
    cats = [
        intake.open_catalog(omsa.CAT_PATH(catalog_name, project_name))
        for catalog_name in cfp.astype(catalog_names, list)
    ]

    # Warning about number of datasets
    ndata = np.sum([len(list(cat)) for cat in cats])
    if ndatasets != -1:
        print(
            f"Note that we are using {ndatasets} datasets of {ndata} datasets. This might take awhile."
        )
    else:
        print(f"Note that there are {ndata} datasets to use. This might take awhile.")

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
                    # import pdb; pdb.set_trace()
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

            # # set model output to UTC
            # tkey = model_var.cf["T"].name
            # model_var[tkey] = model_var[tkey].to_index().tz_localize("UTC")
            # instead turn off time zone for data

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
                print("source name: ", source_name)
                dfd = cat[source_name].read()
                dfd.cf["T"] = pd.to_datetime(dfd.cf["T"])
                dfd.set_index(dfd.cf["T"], inplace=True)
                if dfd.index.tz is not None:
                    warnings.warn(
                        f"Dataset {source_name} had a timezone {dfd.index.tz} which is being removed. Make sure the timezone matches the model output.",
                        RuntimeWarning,
                    )
                    dfd.index = dfd.index.tz_convert(None)

                # import pdb; pdb.set_trace()
                df = omsa.stats._align(dfd.cf[key_variable], model_var)

            # pull out depth at surface?

            # Where to save stats to?
            stats = df.omsa.compute_stats

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
    if CARTOPY_AVAILABLE:
        figname = omsa.PROJ_DIR(project_name) / "map.png"
        omsa.plot.map.plot_map(np.asarray(maps), figname, dam)
    else:
        print("Not plotting map since cartopy is not installed.")
    print(f"Finished analysis. Find plots in {omsa.PROJ_DIR(project_name)}.")
