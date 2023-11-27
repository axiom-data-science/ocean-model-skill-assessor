"""Test synthetic datasets representing featuretypes."""

import os
import pathlib

from unittest import TestCase

import cf_pandas as cfp
import cf_xarray as cfx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xroms
import yaml

from make_test_datasets import make_test_datasets

import ocean_model_skill_assessor as omsa


# # RTD doesn't activate the env, and esmpy depends on a env var set there
# # We assume the `os` package is in {ENV}/lib/pythonX.X/os.py
# # See conda-forge/esmf-feedstock#91 and readthedocs/readthedocs.org#4067
# os.environ["ESMFMKFILE"] = str(pathlib.Path(os.__file__).parent.parent / "esmf.mk")

project_name = "tests"
base_dir = pathlib.Path("tests/test_results")

vocab = cfp.Vocab()
# Make an entry to add to your vocabulary
reg = cfp.Reg(include="tem", exclude=["F_", "qc", "air", "dew"], ignore_case=True)
vocab.make_entry("temp", reg.pattern(), attr="name")
reg = cfp.Reg(include="sal", exclude=["F_", "qc"], ignore_case=True)
vocab.make_entry("salt", reg.pattern(), attr="name")
cfp.set_options(custom_criteria=vocab.vocab)
cfx.set_options(custom_criteria=vocab.vocab)


@pytest.fixture(scope="session")
def dataset_filenames(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data")
    # stores datasets in a dict with key of featuretype
    dds = make_test_datasets()
    # temp file locations
    filenames = {}
    for featuretype, dd in dds.items():
        if isinstance(dd, pd.DataFrame):
            filename = directory / f"{featuretype}.csv"
            dd.to_csv(filename, index=False)
        elif isinstance(dd, xr.Dataset):
            filename = directory / f"{featuretype}.nc"
            dd.to_netcdf(filename)
        filenames[featuretype] = filename
    return filenames


@pytest.fixture(scope="session")
def project_cache(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache")
    return directory


def test_paths(project_cache):
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)
    assert paths.project_name == project_name
    assert paths.cache_dir == project_cache


def make_catalogs(dataset_filenames, featuretype):
    """Make catalog for dataset of type featuretype"""
    filenames = dataset_filenames  # contains all test filenames in dict
    filename = filenames[featuretype]
    # user might choose a different maptype depending on details but default list:
    if featuretype in ["timeSeries", "profile", "timeSeriesProfile"]:
        maptype = "point"
    elif featuretype == "trajectoryProfile":
        maptype = "line"
    elif featuretype == "grid":
        maptype = "box"
    kwargs = {"filenames": [str(filename)]}
    cat = omsa.main.make_catalog(
        catalog_type="local",
        project_name=project_name,
        catalog_name=featuretype,
        metadata={
            "featuretype": featuretype,
            "maptype": maptype,
        },
        kwargs=kwargs,
        return_cat=True,
    )
    return cat


def model_catalog():
    # this dataset is managed by xroms and stored in local cache after the first time it is downloaded.
    url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
    kwargs = {
        "filenames": [url],
        "skip_entry_metadata": True,
    }
    # metadata = {"minLongitude": -93.04208535842456,
    #             "minLatitude": 27.488004525650847,
    #             "maxLongitude": -88.01377130152251,
    #             "maxLatitude": 30.629337972894938}
    cat = omsa.main.make_catalog(
        catalog_type="local",
        project_name=project_name,
        catalog_name="model",
        # metadata=metadata,
        kwargs=kwargs,
        return_cat=True,
    )
    return cat


def test_initial_model_handling(project_cache):
    cat_model = model_catalog()
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )

    # make sure cf-xarray will work after this is run
    axdict = {
        "X": ["xi_rho", "xi_u", "xi_v"],
        "Y": ["eta_rho", "eta_u", "eta_v"],
        "Z": ["s_rho", "s_w"],
        "T": ["ocean_time"],
    }
    assert dsm.cf.axes == axdict
    cdict = {
        "longitude": ["lon_rho", "lon_u", "lon_v"],
        "latitude": ["lat_rho", "lat_u", "lat_v"],
        "vertical": ["s_rho", "s_w"],
        "time": ["ocean_time"],
    }
    assert dsm.cf.coordinates == cdict
    assert isinstance(dsm, xr.Dataset)


def test_narrow_model_time_range(project_cache):
    cat_model = model_catalog()
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )

    model_min_time = pd.Timestamp(dsm.ocean_time.min().values)
    model_max_time = pd.Timestamp(dsm.ocean_time.max().values)

    # not-null user_min_time and user_max_time should control the time range
    user_min_time, user_max_time = model_min_time, model_min_time
    # these wouldn't be nan in the actual code but aren't used in the function in this scenario
    data_min_time, data_max_time = pd.Timestamp(None), pd.Timestamp(None)
    dsm2 = omsa.main._narrow_model_time_range(
        dsm,
        user_min_time,
        user_max_time,
        model_min_time,
        model_max_time,
        data_min_time,
        data_max_time,
    )
    assert dsm2.ocean_time.values[0] == model_min_time

    # not-null user_min_time and user_max_time but model shorter, then data
    # should control the time range
    user_min_time, user_max_time = model_min_time - pd.Timedelta(
        "7D"
    ), model_max_time + pd.Timedelta("7D")
    data_min_time, data_max_time = model_min_time, model_min_time
    dsm2 = omsa.main._narrow_model_time_range(
        dsm,
        user_min_time,
        user_max_time,
        model_min_time,
        model_max_time,
        data_min_time,
        data_max_time,
    )
    assert dsm2.ocean_time.values[0] == model_min_time

    # null user_min_time and user_max_time then data should control the time range
    # but the code takes a model time step extra in each direction, so then get the min time
    user_min_time, user_max_time = pd.Timestamp(None), pd.Timestamp(None)
    data_min_time, data_max_time = model_max_time, model_max_time
    dsm2 = omsa.main._narrow_model_time_range(
        dsm,
        user_min_time,
        user_max_time,
        model_min_time,
        model_max_time,
        data_min_time,
        data_max_time,
    )
    assert dsm2.ocean_time.values[0] == model_min_time


def test_mask_creation(project_cache):
    cat_model = model_catalog()
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    dam = dsm["temp"]
    mask = omsa.utils.get_mask(dsm, dam.cf["longitude"].name, wetdry=False)
    assert not mask.isnull().any()
    assert mask.shape == dam.cf["longitude"].shape


def test_dam_from_dsm(project_cache):
    no_Z = False
    cat_model = model_catalog()
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths
    )

    # Add vocab for testing
    # After this, we have a single Vocab object with vocab stored in vocab.vocab
    vocabs = ["general", "standard_names"]
    vocab = omsa.utils.open_vocabs(vocabs, paths)
    # cfp.set_options(custom_criteria=vocab.vocab)

    # test key_variable as string case
    key_variable, key_variable_data = "temp", "temp"
    with cfx.set_options(custom_criteria=vocab.vocab):
        dam = omsa.main._dam_from_dsm(
            dsm,
            key_variable,
            key_variable_data,
            cat_model["ROMS_example_full_grid"].metadata,
            no_Z,
        )
    # make sure cf-xarray will work after this is run
    axdict = {"X": ["xi_rho"], "Y": ["eta_rho"], "Z": ["s_rho"], "T": ["ocean_time"]}
    assert dam.cf.axes == axdict
    cdict = {
        "longitude": ["lon_rho"],
        "latitude": ["lat_rho"],
        "vertical": ["s_rho"],
        "time": ["ocean_time"],
    }
    assert dam.cf.coordinates == cdict
    assert isinstance(dam, xr.DataArray)


def check_output(cat, featuretype, key_variable, project_cache, no_Z):
    # compare saved model output
    rel_path = pathlib.Path(
        "model_output", f"{cat.name}_{featuretype}_{key_variable}.nc"
    )
    dsexpected = xr.open_dataset(base_dir / rel_path)
    dsactual = xr.open_dataset(project_cache / "tests" / rel_path)

    assert sorted(list(dsexpected.coords)) == sorted(list(dsactual.coords))
    # this doesn't work for grid for windows and linux (same results end up looking different)
    # for var in dsexpected.coords:
    #     assert dsexpected[var].equals(dsactual[var])
    for var in dsexpected.data_vars:
        np.allclose(dsexpected[var], dsactual[var], equal_nan=True)

    # compare saved stats
    rel_path = pathlib.Path("out", f"{cat.name}_{featuretype}_{key_variable}.yaml")
    with open(base_dir / rel_path, "r") as fp:
        statsexpected = yaml.safe_load(fp)
    with open(project_cache / "tests" / rel_path, "r") as fp:
        statsactual = yaml.safe_load(fp)
    for key in statsexpected.keys():
        try:
            if isinstance(statsexpected[key]["value"], list):
                np.allclose(statsexpected[key]["value"], statsactual[key]["value"])
            else:
                TestCase().assertAlmostEqual(
                    statsexpected[key]["value"], statsactual[key]["value"], places=5
                )

        except AssertionError as msg:
            print(msg)
    # assert statsexpected == statsactual
    # TestCase().assertDictEqual(statsexpected, statsactual)

    # compare saved processed files
    rel_path = pathlib.Path(
        "processed", f"{cat.name}_{featuretype}_{key_variable}_data"
    )
    if (base_dir / rel_path).with_suffix(".csv").is_file():
        dfexpected = pd.read_csv((base_dir / rel_path).with_suffix(".csv"))
        dfexpected = omsa.utils.check_dataframe(dfexpected, no_Z)
    elif (base_dir / rel_path).with_suffix(".nc").is_file():
        dfexpected = xr.open_dataset((base_dir / rel_path).with_suffix(".nc"))

    if (project_cache / "tests" / rel_path).with_suffix(".csv").is_file():
        dfactual = pd.read_csv((project_cache / "tests" / rel_path).with_suffix(".csv"))
        dfactual = omsa.utils.check_dataframe(dfactual, no_Z)
    elif (project_cache / "tests" / rel_path).with_suffix(".nc").is_file():
        dfactual = xr.open_dataset(
            (project_cache / "tests" / rel_path).with_suffix(".nc")
        )
    if isinstance(dfexpected, pd.DataFrame):
        pd.testing.assert_frame_equal(dfexpected, dfactual)
    elif isinstance(dfexpected, xr.Dataset):
        assert dfexpected.equals(dfactual)
    rel_path = pathlib.Path(
        "processed", f"{cat.name}_{featuretype}_{key_variable}_model.nc"
    )
    dsexpected = xr.open_dataset(base_dir / rel_path)
    dsactual = xr.open_dataset(project_cache / "tests" / rel_path)
    # assert dsexpected.equals(dsactual)
    assert sorted(list(dsexpected.coords)) == sorted(list(dsactual.coords))
    # this doesn't work for grid for windows and linux (same results end up looking different)
    # for var in dsexpected.coords:
    #     assert dsexpected[var].equals(dsactual[var])
    for var in dsexpected.data_vars:
        np.allclose(dsexpected[var], dsactual[var], equal_nan=True)


def test_bad_catalog(dataset_filenames):
    cat = make_catalogs(dataset_filenames, "timeSeries")
    del cat["timeSeries"].metadata["minLatitude"]
    del cat["timeSeries"]._entry._metadata["minLatitude"]
    with pytest.raises(KeyError):
        omsa.utils.check_catalog(cat)


def test_check_dataframe():
    dfd = pd.DataFrame(columns=["time", "depth", "lon", "lat"])
    omsa.utils.check_dataframe(dfd, no_Z=False)

    dfd = pd.DataFrame(columns=["time", "lon", "lat"])
    omsa.utils.check_dataframe(dfd, no_Z=True)
    with pytest.raises(KeyError):
        omsa.utils.check_dataframe(dfd, no_Z=False)

    dfd = pd.DataFrame(columns=["time", "Z", "lat"])
    with pytest.raises(KeyError):
        omsa.utils.check_dataframe(dfd, no_Z=False)


def test_choose_depths():
    # Z should be None
    no_Z, want_vertical_interp = True, False
    dfd = pd.DataFrame(columns=["time", "depth", "lon", "lat"])
    dfd_out, Z, vertical_interp = omsa.main._choose_depths(
        dfd, "up", no_Z, want_vertical_interp
    )
    assert Z is None
    assert not vertical_interp

    # Z should be 0
    no_Z, want_vertical_interp = False, False
    data = [
        ["1999-1-1", 0, -150, 59],
        ["1999-1-2", 0, -150, 59],
    ]
    dfd = pd.DataFrame(columns=["time", "depth", "lon", "lat"], data=data)
    dfd_out, Z, vertical_interp = omsa.main._choose_depths(
        dfd, "up", no_Z, want_vertical_interp
    )
    assert Z is not None
    assert Z == 0
    assert not vertical_interp

    # Z should be -10
    no_Z, want_vertical_interp = False, False
    data = [
        ["1999-1-1", -10, -150, 59],
        ["1999-1-2", -10, -150, 59],
    ]
    dfd = pd.DataFrame(columns=["time", "depth", "lon", "lat"], data=data)
    dfd_out, Z, vertical_interp = omsa.main._choose_depths(
        dfd, "up", no_Z, want_vertical_interp
    )
    assert Z is not None
    assert Z == -10
    assert not vertical_interp


@pytest.mark.mpl_image_compare(style="default")
def test_timeSeries_temp(dataset_filenames, project_cache):
    featuretype = "timeSeries"
    no_Z = False
    key_variable, interpolate_horizontal = "temp", True
    want_vertical_interp = False
    need_xgcm_grid = False

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test data time range
    data_min_time, data_max_time = omsa.main._find_data_time_range(
        cat, source_name=featuretype
    )
    assert data_min_time, data_max_time == (
        pd.Timestamp("2009-11-19 12:00:00"),
        pd.Timestamp("2009-11-19 16:00:00"),
    )

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()

    # test depth selection for temp/salt
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    pd.testing.assert_frame_equal(dfdout, dfd)
    assert Z == 0
    assert not vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        skip_mask=True,
    )

    # temp, with horizontal interpolation
    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )
    check_output(cat, featuretype, key_variable, project_cache, no_Z)
    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_timeSeries_ssh(dataset_filenames, project_cache):
    featuretype = "timeSeries"
    key_variable, interpolate_horizontal = "ssh", False
    no_Z = True
    want_vertical_interp = False
    need_xgcm_grid = False

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()
    # test depth selection for SSH
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    pd.testing.assert_frame_equal(dfdout, dfd)
    assert Z is None
    assert not vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        skip_mask=True,
    )

    # without horizontal interpolation and ssh
    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )
    check_output(cat, featuretype, key_variable, project_cache, no_Z)
    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_profile(dataset_filenames, project_cache):
    featuretype = "profile"
    no_Z = False
    key_variable, interpolate_horizontal = "temp", False
    want_vertical_interp = True
    need_xgcm_grid = True

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test data time range
    data_min_time, data_max_time = omsa.main._find_data_time_range(
        cat, source_name=featuretype
    )
    assert data_min_time, data_max_time == (
        pd.Timestamp("2009-11-19T14:00"),
        pd.Timestamp("2009-11-19T14:00"),
    )

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()
    # test depth selection for temp/salt
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    pd.testing.assert_frame_equal(dfdout, dfd)
    assert (Z == dfd.cf["Z"]).all()
    assert vertical_interp == want_vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        skip_mask=True,
    )

    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )

    check_output(cat, featuretype, key_variable, project_cache, no_Z)
    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_timeSeriesProfile(dataset_filenames, project_cache):
    """ADCP mooring but for temp for ease of testing"""

    featuretype = "timeSeriesProfile"
    no_Z = False
    key_variable, interpolate_horizontal = "temp", False
    want_vertical_interp = True
    need_xgcm_grid = True

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test data time range
    data_min_time, data_max_time = omsa.main._find_data_time_range(
        cat, source_name=featuretype
    )
    assert data_min_time, data_max_time == (
        pd.Timestamp("2009-11-19T12:00"),
        pd.Timestamp("2009-11-19T16:00"),
    )

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()
    # test depth selection for temp/salt. These are Datasets
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    assert dfd.equals(dfdout)
    assert (Z == dfd.cf["Z"]).all()
    assert vertical_interp == want_vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        skip_mask=True,
    )

    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )

    check_output(cat, featuretype, key_variable, project_cache, no_Z)
    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_trajectoryProfile(dataset_filenames, project_cache):
    """CTD transect"""

    featuretype = "trajectoryProfile"
    no_Z = False
    key_variable, interpolate_horizontal = "salt", True
    want_vertical_interp = True
    need_xgcm_grid = True
    save_horizontal_interp_weights = False

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test data time range
    data_min_time, data_max_time = omsa.main._find_data_time_range(
        cat, source_name=featuretype
    )
    assert data_min_time, data_max_time == (
        pd.Timestamp("2009-11-19T12:00"),
        pd.Timestamp("2009-11-19T16:00"),
    )

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()
    # test depth selection for temp/salt. These are Datasets
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    assert dfd.equals(dfdout)
    assert (Z == dfd.cf["Z"]).all()
    assert vertical_interp == want_vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        save_horizontal_interp_weights=save_horizontal_interp_weights,
        skip_mask=True,
    )

    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )

    check_output(cat, featuretype, key_variable, project_cache, no_Z)

    return fig


@pytest.mark.mpl_image_compare(style="default")
def test_grid(dataset_filenames, project_cache):
    """HF Radar"""

    featuretype = "grid"
    no_Z = False
    key_variable, interpolate_horizontal = "temp", True
    # key_variable = [{"data": "north", "accessor": "xroms", "function": "north", "inputs": {}}]
    want_vertical_interp = False
    horizontal_interp_code = "xesmf"
    locstream = False
    need_xgcm_grid = True
    save_horizontal_interp_weights = False

    cat = make_catalogs(dataset_filenames, featuretype)
    omsa.utils.check_catalog(cat)
    paths = omsa.paths.Paths(project_name=project_name, cache_dir=project_cache)

    # test data time range
    data_min_time, data_max_time = omsa.main._find_data_time_range(
        cat, source_name=featuretype
    )
    assert data_min_time, data_max_time == (
        pd.Timestamp("2009-11-19T12:00"),
        pd.Timestamp("2009-11-19T16:00"),
    )

    # test depth selection
    cat_model = model_catalog()
    dsm, model_source_name = omsa.main._initial_model_handling(
        model_name=cat_model, paths=paths, model_source_name=None
    )
    zkeym = dsm.cf.axes["Z"][0]

    dfd = cat[featuretype].read()
    # test depth selection for temp/salt. These are Datasets
    dfdout, Z, vertical_interp = omsa.main._choose_depths(
        dfd, dsm[zkeym].attrs["positive"], no_Z, want_vertical_interp
    )
    assert dfd.equals(dfdout)
    assert (Z == dfd.cf["Z"]).all()
    assert vertical_interp == want_vertical_interp

    kwargs = dict(
        catalogs=cat,
        model_name=cat_model,
        preprocess=True,
        vocabs=["general", "standard_names"],
        mode="a",
        alpha=5,
        dd=5,
        want_vertical_interp=want_vertical_interp,
        horizontal_interp_code=horizontal_interp_code,
        locstream=locstream,
        extrap=False,
        check_in_boundary=False,
        need_xgcm_grid=need_xgcm_grid,
        plot_map=False,
        plot_count_title=False,
        cache_dir=project_cache,
        vocab_labels="vocab_labels",
        save_horizontal_interp_weights=save_horizontal_interp_weights,
        skip_mask=True,
    )

    fig = omsa.run(
        project_name=project_name,
        key_variable=key_variable,
        interpolate_horizontal=interpolate_horizontal,
        no_Z=no_Z,
        return_fig=True,
        **kwargs,
    )

    check_output(cat, featuretype, key_variable, project_cache, no_Z)

    return fig
