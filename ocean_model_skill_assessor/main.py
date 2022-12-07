"""
Main run functions.
"""

import pathlib
from typing import DefaultDict, Dict, Optional, Sequence, Union
import warnings
import cf_xarray
import cf_pandas as cfp
import extract_model as em
import intake
import numpy as np
# import ocean_data_gateway as odg
import pandas as pd
import xarray as xr

import ocean_model_skill_assessor as omsa

from intake.catalog.local import LocalCatalogEntry
from intake.catalog import Catalog

def make_kw(bbox, time_range):
    """Make kw for search.

    Parameters
    ----------
    bbox: list
        Geographic bounding box: [min_lon, min_lat, max_lon, max_lat]
    time_range: list
        [start_time, end_time] where each are strings that can be interpreted
        with pandas `Timestamp`.

    Returns
    -------
    Dictionary of parameters for search.
    """

    keys = ["min_lon", "min_lat", "max_lon", "max_lat", "min_time", "max_time"]

    kw = {key: value for key, value in zip(keys, bbox + time_range)}

    return kw


def find_bbox(ds):
    """Determine bounds and boundary of model.

    Parameters
    ----------
    ds: Dataset
        xarray Dataset containing model output.

    Returns
    -------
    List containing geographic bounding box of model output: [min_lon, min_lat, max_lon, max_lat] and Nx2 array of boundary of model.
    """

    try:
        lon = ds.cf["longitude"].values
        lat = ds.cf["latitude"].values

    except KeyError:
        # In case there are multiple grids, just take first one;
        # they are close enough
        lon = ds[ds.cf.coordinates["longitude"][0]]
        lat = ds[ds.cf.coordinates["latitude"][0]]
        # lon = list(ds.cf[["longitude"]].coords.keys())[0].values
        # lat = list(ds.cf[["latitude"]].coords.keys())[0].values

    min_lon = lon.min()
    max_lon = lon.max()
    min_lat = lat.min()
    max_lat = lat.max()
    #     min_lon = float(ds[lon].min())
    #     max_lon = float(ds[lon].max())
    #     min_lat = float(ds[lat].min())
    #     max_lat = float(ds[lat].max())
    #     import pdb; pdb.set_trace()
    if lon.ndim == 2:
        lonb = np.concatenate((lon[:, 0], lon[-1, :], lon[::-1, -1], lon[0, ::-1]))
        latb = np.concatenate((lat[:, 0], lat[-1, :], lat[::-1, -1], lat[0, ::-1]))
    elif lon.ndim == 1:
        nlon, nlat = ds["lon"].size, ds["lat"].size
        lonb = np.concatenate(([lon[0]] * nlat, lon[:], [lon[-1]] * nlat, lon[::-1]))
        latb = np.concatenate((lat[:], [lat[-1]] * nlon, lat[::-1], [lat[0]] * nlon))
    boundary = np.vstack((lonb, latb)).T

    return [min_lon, min_lat, max_lon, max_lat], boundary


def read_model(loc_model, xarray_kwargs, time_range=None):
    """Read in model output input by user.

    Parameters
    ----------
    loc_model : str
        Relative or absolute, local or nonlocal path to model output.
    xarray_kwargs : dict, optional
        Keyword arguments to pass into `xr.open_dataset`.
    time_range: list
        [min_time, max_time] for desired time range of search where each
        are strings that can be interpreted with pandas `Timestamp`.

    Returns
    -------
    xarray Dataset containing model output.
    """

    if isinstance(loc_model, list):
        dsm = xr.open_mfdataset(loc_model, **xarray_kwargs)
    else:
        dsm = xr.open_dataset(loc_model, **xarray_kwargs)

    # add more cf-xarray info
    dsm = dsm.cf.guess_coord_axis()

    # drop duplicate time indices if present
    # also limit the time range of the model output to what we are requesting from the data to
    # not waste extra time on the model interpolation
    # https://stackoverflow.com/questions/51058379/drop-duplicate-times-in-xarray
    _, index = np.unique(dsm.cf["T"], return_index=True)

    if time_range:
        dsm = dsm.cf.isel(T=index).cf.sel(T=slice(time_range[0], time_range[1]))

    # force longitude to be from -180 to 180
    for lkey in dsm.cf.coordinates["longitude"]:
        dsm[lkey] = dsm[lkey].where(dsm[lkey] < 180, dsm[lkey] - 360)
    # lkey = dsm.cf["longitude"].name
    # dsm[lkey] = dsm.cf["longitude"].where(
    #     dsm.cf["longitude"] < 180, dsm.cf["longitude"] - 360
    # )

    return dsm


def prep_plot(search):
    """Put together inputs for map plot."""

    sub = search.meta.loc[
        search.dataset_ids,
        [
            "geospatial_lon_min",
            "geospatial_lat_min",
            "geospatial_lon_max",
            "geospatial_lat_max",
        ],
    ]
    lls = sub.values
    names = list(sub.index.values)

    # put out stationary data
    istations = lls[:, 0] == lls[:, 2]

    # temporarily remove dataset_ids that aren't a station
    ids_to_remove = list(np.array(names)[~istations])
    for id_remove in ids_to_remove:
        ind = search.sources[0].dataset_ids.index(id_remove)
        search.sources[0].dataset_ids.pop(ind)

    sub = search.meta.loc[
        search.dataset_ids,
        [
            "geospatial_lon_min",
            "geospatial_lat_min",
            "geospatial_lon_max",
            "geospatial_lat_max",
        ],
    ]
    lls = sub.values
    names = list(sub.index.values)

    # put out stationary data
    istations = lls[:, 0] == lls[:, 2]
    lls_stations = lls[istations, :2]
    names_stations = list(np.array(names)[istations])
    if len(names_stations) == 0:
        names_stations = None
        lls_stations = None

    # pull out data over range
    lls_box = lls[~istations]
    names_boxes = list(np.array(names)[~istations])

    if len(names_boxes) == 0:
        names_boxes = None
        lls_box = None

    return lls_stations, names_stations, lls_box, names_boxes


def prep_em(input_data):
    """Prepare to run extract_model."""

    if isinstance(input_data, pd.DataFrame):
        data = input_data
        tname = data.cf["T"].name
        data[tname] = pd.to_datetime(data.cf["T"])
        data = data.set_index(data.cf["T"])
    else:
        data = input_data
    lon = float(data.cf["longitude"].values[0])
    lat = float(data.cf["latitude"].values[0])
    T = None
    # only compare surface
    Z = None

    return data, lon, lat, T, Z


def make_local_catalog(filenames: Optional[Union[Sequence,str]] = None,) -> Catalog:
    """_summary_

    Parameters
    ----------
    filenames : Optional[Union[Sequence,str]], optional
        _description_, by default None

    Returns
    -------
    Catalog
        _description_
    """
    import mimetypes
    sources = []
    for filename in filenames:
        if "csv" in mimetypes.guess_type(filename)[0]:
            sources.append(getattr(intake, "open_csv")(filename))
        elif "netcdf" in mimetypes.guess_type(filename)[0]:
            sources.append(getattr(intake, "open_netcdf")(filename))
            
    from intake.catalog.local import LocalCatalogEntry
    from intake.catalog import Catalog

    # create dictionary of catalog entries
    entries = {
        f"source{i}": LocalCatalogEntry(
            name=f"source{i}",
            description=source.description,
            driver=source._yaml()['sources'][source.name]['driver'],
            args=source._yaml()["sources"][source.name]["args"],
            metadata=source.metadata,
        )
        for i, source in enumerate(sources)
    }

    # create catalog
    cat = Catalog.from_dict(
        entries,
        name="Input files",
        description="full_cat_description",
        metadata="full_cat_metadata",
    )
    return cat


def make_catalog(catalog_type: str,
                 project_name: str,
                 catalog_name: Optional[str] = None,
                #  nickname: Optional[str] = None,
                #  filenames: Optional[Union[Sequence,str]] = None,
                #  erddap_server: Optional[str] = None,
                #  axds_type: Optional[str] = "platform2",
                 kwargs: Dict[str, Union[str, int, float]] = None,
                 kwargs_search: Dict[str, Union[str, int, float]] = None,
                 vocab: Optional[Union[DefaultDict[str, Dict[str, str]],str,pathlib.PurePath]] = None,
                #  page_size: int = 10,
                 return_cat = True,
                 save_cat = False,
                 ):
    """Make a catalog given input selections.

    Parameters
    ----------
    catalog_type : str
        Which type of catalog to make? Options are "erddap", "axds", or "local".
    project_name : str
        Subdirectory in cache dir to store files associated together.
    catalog_name : str
        Catalog name, with or without suffix of yaml.
    nickname : str
        Variable nickname representing which variable in vocabulary you are searching for.
    
    kwargs : 
        All keyword arguments for the given catalog.
        * axds:
        
          * datatype: default "platform2"
          * page_size: default 10
          * keys_to_match: Optional[Union[str, list]] = None,
          * standard_names: Optional[Union[str, list]] = None,
          * verbose: bool = False,
          * name: str = "catalog",
          * description: str = "Catalog of Axiom assets.",
          * metadata: dict = None,
          * ttl: Optional[int] = None,
        
        * erddap:
          * erddap_server : Optional[str], optional
          
        * local
          * filenames : Optional[Union[Sequence,str]], optional

    kwargs_search : Dict[str, Union[str, int, float]], optional
        _description_, by default None
    vocab : Optional[DefaultDict[str, Dict[str, str]]], optional
        _description_, by default None
    """
    
    if kwargs is None:
        kwargs = {}

    # Should I require vocab if nickname is not None?
    # if vocab is None:
    #     # READ IN DEFAULT AND SET VOCAB
    #     vocab = cfp.Vocab("vocabs/general")
        
    # elif isinstance(vocab, str):
    #     vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))

    if isinstance(vocab, str):
        vocab = cfp.Vocab(omsa.VOCAB_PATH(vocab))
    
    # # Can use filenames OR erddap_server OR axds_type
    # if [(filenames is not None), (erddap_server is not None), (axds_type is not None)].count(True) > 1:
    #     raise KeyError("Input `filenames` or `erddap_server` or `axds_type` but not more than one.")

    if catalog_type == "local":
        if "filenames" not in kwargs:
            raise ValueError("For `catalog_type=='local'`, must input `filenames`.")
        cat = make_local_catalog(kwargs["filenames"])

    elif catalog_type == "erddap":
        if "erddap_server" not in kwargs:
            raise ValueError("For `catalog_type=='erddap'`, must input `erddap_server`.")
        if vocab is not None:
            with cfp.set_options(custom_criteria=vocab.vocab):
                cat = intake.open_erddap_cat(kwargs["erddap_server"], kwargs_search=kwargs_search, category_search=["standard_name", nickname])
        else:
            cat = intake.open_erddap_cat(kwargs["erddap_server"], kwargs_search=kwargs_search)
        catalog_name = "erddap_cat" if catalog_name is None else catalog_name
        
    elif catalog_type == "axds":
        catalog_name = "axds_cat" if catalog_name is None else catalog_name
        kwargs["name"] = catalog_name
        if vocab is not None:
            with cfp.set_options(custom_criteria=vocab.vocab):
                cat = intake.open_axds_cat(kwargs_search=kwargs_search, **kwargs)
        else:
            cat = intake.open_axds_cat(kwargs_search=kwargs_search, **kwargs)

    if save_cat:
        # save cat to file
        cat.save(omsa.CAT_PATH(catalog_name, project_name))
        print(f"Catalog saved to {omsa.CAT_PATH(catalog_name, project_name)}.")

    if return_cat:
        return cat
    
    
def run2(catalog_paths: Union[Sequence,str,pathlib.PurePath],
         nickname: str,
         model_url: str,
         project_name: Optional[str] = None,
         vocab: Optional[DefaultDict[str, Dict[str, str]]] = None,
         ):
    """_summary_

    Parameters
    ----------
    catalog_paths : Union[Sequence[str,pathlib.PurePath],str,pathlib.PurePath]
        _description_
    nickname : str
        _description_
    model_url : str
        _description_
    project_name : str, optional
        If not input, will use the the parent to catalog_paths as the project directory, but can be overridden by inputting this keyword.
    vocab : Optional[DefaultDict[str, Dict[str, str]]], optional
        _description_, by default None
    """
    
    if vocab is None:
        # READ IN DEFAULT AND SET VOCAB
        vocab = cfp.Vocab("vocabs/general")
        
    if project_name is None:
        project_name is cfp.astype(catalog_paths, list)[0].parent
        
    # read in model output
    dsm = xr.open_mfdataset(model_url, em.preprocess) if cfp.astype(model_url, list) else xr.open_dataset(model_url, em.preprocess)
    
    # use only one variable from model
    dam = dsm.cf[nickname]

    cats = [intake.open_catalog(catalog_path) for catalog_path in catalog_paths]

    # loop over catalogs and sources to pull out lon/lat locations for plot
    maps = []
    for cat in cats:
        for source_name in list(cat):
            min_lon, max_lon = cat[source_name].metadata["min_lon"], cat[source_name].metadata["max_lon"]
            min_lat, max_lat = cat[source_name].metadata["min_lat"], cat[source_name].metadata["max_lat"]
            min_time, max_time = cat[source_name].metadata["min_time"], cat[source_name].metadata["max_time"]
            maps.append([min_lon, max_lon, min_lat, max_lat, source_name])
            
            if min_lon != max_lon or min_lat != max_lat:
                warnings.warn(f"Source {source_name} in catalog {cat.name} is not stationary so not plotting.")
                continue
            
            # Pull out nearest model output to data
            # use extract_model
            kwargs = dict(
                longitude=min_lon,
                latitude=min_lat,
                iT = slice(min_time, max_time),
                # T=cat[source_name],
                Z=0,
                method="nearest",
            )
            # if T is not None:
            #     kwargs["T"] = T

            # xoak doesn't work for 1D lon/lat coords
            if (
                dam.cf["longitude"].ndim
                == dam.cf["latitude"].ndim
                == 1
            ):
                model_var = dam.cf.sel(**kwargs)#.to_dataset()

            elif (
                dam.cf["longitude"].ndim
                == dam.cf["latitude"].ndim
                == 2
            ):
                model_var = dam.em.sel2dcf(**kwargs)#.to_dataset()
        
            # Combine and align the two time series of variable
            df = omsa.stats._align(cat[source_name].to_dask().cf[nickname], model_var)#.cf[variable])

            # pull out depth at surface

            # df = omsa.stats._align(data.cf[variable], model_var)#.cf[variable])
            # Where to save stats to?
            stats = df.omsa.compute_stats

            # Write stats on plot
            figname = f"{source_name}_{nickname}.png"
            df.omsa.plot(
                title=f"{source_name}",
                ylabel=dam.name,
                figname=figname_data_prefix + figname,
                stats=stats,
            )
    
    # map of model domain with data locations
    plot_map(maps, project_name)
    


# def run(
#     loc_model,
#     axds=None,
#     bbox=None,
#     criteria=None,
#     erddap=None,
#     figname_map=None,
#     figname_data_prefix="",
#     horizontal_interp=False,
#     local=None,
#     only_search=False,
#     only_searchplot=False,
#     parallel=True,
#     proj=None,
#     readers=None,
#     run_qc=False,
#     skip_units=False,
#     stations=None,
#     time_range=None,
#     variables=None,
#     var_def=None,
#     xarray_kwargs=None,
# ):
#     """Run package.

#     Parameters
#     ----------
#     loc_model : str
#         Relative or absolute, local or nonlocal path to model output.
#     axds : dict, optional
#         Inputs for axds reader.
#     bbox : list, optional
#         [min_lon, min_lat, max_lon, max_lat] if you want to override
#         the default of taking the model bounding box for your region
#         search.
#     criteria : dict, str, optional
#         Regex criteria to use for identifying variables by name or attributes. Note that this will both be used in this package and passed on to odg.
#     erddap : dict, optional
#         Inputs for ERDDAP reader.
#     figname_map : str, optional
#         Figure name for map showing data locations.
#     figname_data_prefix : str, optional
#         Prefix for figures for dataset-model comparisons.
#     horizontal_interp : bool, optional
#         If True, use `em.select()` to interpolate to the data location horizontally.
#         If False, use `em.sel2d()` to use the nearest grid point to the data location.
#     local : dict, optional
#         Inputs for local reader.
#     only_search : boolean, optional
#         Stop after search is initiated.
#     only_searchplot : boolean, optional
#         Stop after search and map plot is perform.
#     parallel : boolean, optional
#         Whether to run in parallel with `multiprocessing` library where
#         possible. Default is True.
#     proj: proj instance
#         Projection from cartopy. Example: `cartopy.crs.Mercator()`.
#     readers : odg reader or list of readers, optional
#         Can specify which of the available readers to use in your search.
#         Options are odg.erddap, odg.axds, and odg.local. Default is to use all.
#     run_qc : boolean, optional
#         If True, run basic QC.
#     skip_units : boolean, optional
#         If True, assume units are the same between model output and datasets.
#     stations : str, list, optional
#         Stations or dataset_ids for `approach=='stations'`.
#     time_range: list
#         [min_time, max_time] for desired time range of search where each
#         are strings that can be interpreted with pandas `Timestamp`.
#     var_def : dict, optional
#         Variable units and QARTOD information. Necessary for QC. Variables key nicknames must match those in `criteria`.
#     variables : str, list, optional
#         Variables to search for.
#     xarray_kwargs : dict, optional
#         Keyword arguments to pass into `xr.open_dataset`.

#     Returns
#     -------
#     An `ocean_data_gateway` Gateway object.
#     """

#     if xarray_kwargs is None:
#         xarray_kwargs = {}

#     # Set custom criteria
#     if criteria:
#         if isinstance(criteria, str) and criteria[:4] == "http":
#             criteria = odg.return_response(criteria)
#         cf_xarray.set_options(custom_criteria=criteria)

#     if var_def:
#         if isinstance(var_def, str) and var_def[:4] == "http":
#             var_def = odg.return_response(var_def)

#     dsm = read_model(loc_model, xarray_kwargs, time_range)

#     # Start set up for kwargs for search
#     kwargs = dict(
#         criteria=criteria,
#         var_def=var_def,
#         parallel=parallel,
#         variables=variables,
#         readers=readers,
#         local=local,
#         erddap=erddap,
#         axds=axds,
#         skip_units=skip_units,
#     )

#     bbox_model, boundary = find_bbox(dsm)
#     if bbox is None:
#         bbox = bbox_model

#     # if approach == "region":

#     #     # Require time_range
#     #     assert time_range, "Require time range for `approach=='region'`."

#     #     kw = make_kw(bbox, time_range)

#     #     kwargs["kw"] = kw

#     # elif (approach == "stations") and time_range:

#     #     kw = dict(min_time=time_range[0], max_time=time_range[1])

#     #     kwargs["kw"] = kw
#     #     kwargs["stations"] = stations

#     # Perform search
#     search = odg.Gateway(**kwargs)

#     # return if no datasets discovered
#     if len(search.dataset_ids) == 0:
#         print("No dataset_ids found. Try a different search.")
#         return search
#     if only_search:
#         return search

#     # Plot discovered datasets
#     lls_stations, names_stations, lls_box, names_boxes = prep_plot(search)
#     # import pdb; pdb.set_trace()
#     omsa.plots.map.plot(
#         lls_stations=lls_stations,
#         names_stations=names_stations,
#         lls_boxes=lls_box,
#         names_boxes=names_boxes,
#         boundary=boundary,
#         res="10m",
#         figname=figname_map,
#         proj=proj,
#     )

#     if only_searchplot:
#         print("Searched and plotted.")
#         return search

#     # data locations to calculate model at
#     for dataset_id in search.dataset_ids:

#         # Run QC
#         # Results not currently incorporated into rest of analysis.
#         if run_qc:
#             obs = search.qc(
#                 dataset_ids=dataset_id, verbose=False, skip_units=skip_units
#             )

#         data, lon, lat, T, Z = prep_em(search[dataset_id])
#         if data is None:
#             continue

#         for variable in variables:

#             if horizontal_interp:
#                 kwargs = dict(
#                     da=dsm.cf[variable]
#                     .cf.isel(Z=0)
#                     .cf.sel(lon=slice(lon - 5, lon + 5), lat=slice(lat - 5, lat + 5)),
#                     longitude=lon,
#                     latitude=lat,
#                     T=T,
#                     iZ=Z,
#                     locstream=True,
#                 )

#                 model_var = em.select(**kwargs).to_dataset()

#             else:
#                 kwargs = dict(
#                     longitude=lon,
#                     latitude=lat,
#                     #                     T=T,
#                     Z=0,
#                     method="nearest",
#                 )
#                 if T is not None:
#                     kwargs["T"] = T

#                 # xoak doesn't work for 1D lon/lat coords
#                 if (
#                     dsm.cf[variable].cf["longitude"].ndim
#                     == dsm.cf[variable].cf["latitude"].ndim
#                     == 1
#                 ):
#                     model_var = dsm.cf[variable].cf.sel(**kwargs).to_dataset()

#                 elif (
#                     dsm.cf[variable].cf["longitude"].ndim
#                     == dsm.cf[variable].cf["latitude"].ndim
#                     == 2
#                 ):
#                     model_var = dsm.cf[variable].em.sel2dcf(**kwargs).to_dataset()

#             # Combine and align the two time series of variable
#             df = omsa.stats._align(data.cf[variable], model_var.cf[variable])
#             stats = df.omsa.compute_stats

#             # Write stats on plot
#             longname = dsm.cf[variable].attrs["long_name"]
#             ylabel = f"{longname}"
#             figname = f"{dataset_id}_{variable}.png"
#             df.omsa.plot(
#                 title=f"{dataset_id}",
#                 ylabel=ylabel,
#                 figname=figname_data_prefix + figname,
#                 stats=stats,
#             )

#     return search
