"""
Main run functions.
"""

from pathlib import Path

import cf_xarray
import extract_model as em
import numpy as np
import ocean_data_gateway as odg
import pandas as pd
import xarray as xr

import ocean_model_skill_assessor as omsa


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
    except KeyError as e:
        # In case there are multiple grids, just take first one;
        # they are close enough
        lon = list(ds.cf[["longitude"]].coords.keys())[0].values
        lat = list(ds.cf[["latitude"]].coords.keys())[0].values

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
    lkey = dsm.cf["longitude"].name
    dsm[lkey] = dsm.cf["longitude"].where(
        dsm.cf["longitude"] < 180, dsm.cf["longitude"] - 360
    )

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


def run(
    approach,
    loc_model,
    axds=None,
    bbox=None,
    criteria=None,
    erddap=None,
    figname_map=None,
    figname_data_prefix="",
    local=None,
    only_search=False,
    only_searchplot=False,
    output_dir=None,
    parallel=True,
    readers=None,
    run_qc=False,
    skip_units=False,
    stations=None,
    time_range=None,
    variables=None,
    var_def=None,
    xarray_kwargs=None,
):
    """Run package.

    Parameters
    ----------
    approach : str
        'region' or 'stations'
    loc_model : str
        Relative or absolute, local or nonlocal path to model output.
    axds : dict, optional
        Inputs for axds reader.
    bbox : list, optional
        [min_lon, min_lat, max_lon, max_lat] if you want to override
        the default of taking the model bounding box for your region
        search.
    criteria : dict, str, optional
        Regex criteria to use for identifying variables by name or attributes. Note that this will both be used in this package and passed on to odg.
    erddap : dict, optional
        Inputs for ERDDAP reader.
    figname_map : str, optional
        Figure name for map showing data locations.
    figname_data_prefix : str, optional
        Prefix for figures for dataset-model comparisons.
    local : dict, optional
        Inputs for local reader.
    only_search : boolean, optional
        Stop after search is initiated.
    only_searchplot : boolean, optional
        Stop after search and map plot is perform.
    output_dir: str, optional
        Path to directory where output files will be saved.  Otherwise, files will be saved in the current directory.
    parallel : boolean, optional
        Whether to run in parallel with `multiprocessing` library where
        possible. Default is True.
    readers : odg reader or list of readers, optional
        Can specify which of the available readers to use in your search.
        Options are odg.erddap, odg.axds, and odg.local. Default is to use all.
    run_qc : boolean, optional
        If True, run basic QC.
    skip_units : boolean, optional
        If True, assume units are the same between model output and datasets.
    stations : str, list, optional
        Stations or dataset_ids for `approach=='stations'`.
    time_range: list
        [min_time, max_time] for desired time range of search where each
        are strings that can be interpreted with pandas `Timestamp`.
    var_def : dict, optional
        Variable units and QARTOD information. Necessary for QC. Variables key nicknames must match those in `criteria`.
    variables : str, list, optional
        Variables to search for.
    xarray_kwargs : dict, optional
        Keyword arguments to pass into `xr.open_dataset`.

    Returns
    -------
    An `ocean_data_gateway` Gateway object.
    """

    # Prepare output directory
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare stats output
    stats_summary = omsa.stats.StatsSummary(approach, loc_model)
    stats_summary_fpath = output_dir / "stats_summary.csv"

    if xarray_kwargs is None:
        xarray_kwargs = {}

    # Set custom criteria
    if criteria:
        if isinstance(criteria, str) and criteria[:4] == "http":
            criteria = odg.return_response(criteria)
        cf_xarray.set_options(custom_criteria=criteria)

    if var_def:
        if isinstance(var_def, str) and var_def[:4] == "http":
            var_def = odg.return_response(var_def)

    dsm = read_model(loc_model, xarray_kwargs, time_range)

    # Start set up for kwargs for search
    kwargs = dict(
        criteria=criteria,
        var_def=var_def,
        approach=approach,
        parallel=parallel,
        variables=variables,
        readers=readers,
        local=local,
        erddap=erddap,
        axds=axds,
        skip_units=skip_units,
    )

    bbox_model, boundary = find_bbox(dsm)
    if bbox is None:
        bbox = bbox_model

    if approach == "region":

        # Require time_range
        assert time_range, "Require time range for `approach=='region'`."

        kw = make_kw(bbox, time_range)

        kwargs["kw"] = kw

    elif (approach == "stations") and time_range:

        kw = dict(min_time=time_range[0], max_time=time_range[1])

        kwargs["kw"] = kw
        kwargs["stations"] = stations

    # Perform search
    search = odg.Gateway(**kwargs)

    # return if no datasets discovered
    if len(search.dataset_ids) == 0:
        print("No dataset_ids found. Try a different search.")
        return search
    if only_search:
        return search

    # Plot discovered datasets
    lls_stations, names_stations, lls_box, names_boxes = prep_plot(search)
    fig_fpath = output_dir / figname_map
    omsa.map.plot(
        lls_stations=lls_stations,
        names_stations=names_stations,
        lls_boxes=lls_box,
        names_boxes=names_boxes,
        boundary=boundary,
        res="10m",
        figname=fig_fpath
    )

    if only_searchplot:
        print("Searched and plotted.")
        return search

    # data locations to calculate model at
    for dataset_id in search.dataset_ids:

        # Run QC
        # Results not currently incorporated into rest of analysis.
        if run_qc:
            obs = search.qc(
                dataset_ids=dataset_id, verbose=False, skip_units=skip_units
            )

        data, lon, lat, T, Z = prep_em(search[dataset_id])
        if data is None:
            continue

        for variable in variables:
            kwargs = dict(
                da=dsm.cf[variable]
                .cf.isel(Z=0)
                .cf.sel(lon=slice(lon - 5, lon + 5), lat=slice(lat - 5, lat + 5)),
                longitude=lon,
                latitude=lat,
                T=T,
                iZ=Z,
                locstream=True,
            )

            model_var = em.select(**kwargs).to_dataset()
            # Combine and align the two time series of variable
            df = omsa.stats._align(data.cf[variable], model_var.cf[variable])
            stats = df.omsa.compute_stats
            stats_summary.add_dataset(dataset_id, variable, stats)

            # Write stats on plot
            longname = dsm.cf[variable].attrs["long_name"]
            ylabel = f"{longname}"
            figname = f"{figname_data_prefix}_{dataset_id}_{variable}.png"
            fig_fpath = output_dir / figname
            df.omsa.plot(
                title=f"{dataset_id}",
                ylabel=ylabel,
                figname=fig_fpath,
                stats=stats,
            )

    stats_summary.to_csv(stats_summary_fpath)

    return search
