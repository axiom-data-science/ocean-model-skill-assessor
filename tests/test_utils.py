import pathlib

from unittest import TestCase, mock

import cf_pandas
import intake_xarray
import numpy as np
import pytest
import shapely.geometry
import xarray as xr

from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry

import ocean_model_skill_assessor as omsa


ds = xr.Dataset()
ds["time"] = (
    "time",
    np.arange(10),
    {"standard_name": "time"},
)
lon, lat = np.arange(10), np.arange(10)
ds["lat"] = (
    ("lat"),
    lat,
    {"units": "degrees_north", "standard_name": "latitude"},
)
ds["lon"] = (
    ("lon"),
    lon,
    {"units": "degrees_east", "standard_name": "longitude"},
)
# lon, lat = np.meshgrid(np.arange(10), np.arange(10))
# ds["lat"] = (
#     ("y", "x"),
#     lat,
#     {"units": "degrees_north", "standard_name": "latitude"},
# )
# ds["lon"] = (
#     ("y","x"),
#     lon,
#     {"units": "degrees_east", "standard_name": "longitude"},
# )


@pytest.fixture(scope="session")
def project_cache(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache")
    return directory


@mock.patch("intake_xarray.base.DataSourceMixin.to_dask")
@mock.patch("intake.open_catalog")
def test_kwargs_search_from_model(mock_open_cat, mock_to_dask, project_cache):

    kwargs_search = {"model_name": "path", "project_name": "test_project"}

    entries = {
        "name": LocalCatalogEntry(
            name="name",
            description="description",
            driver=intake_xarray.opendap.OpenDapSource,
            args={"urlpath": "path", "engine": "netcdf4"},
            metadata={},
            direct_access="allow",
        ),
    }
    cat = Catalog.from_dict(
        entries,
        name="name",
        description="description",
        metadata={},
    )

    mock_open_cat.return_value = cat

    mock_to_dask.return_value = ds

    paths = omsa.paths.Paths(project_name="projectA", cache_dir=project_cache)
    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search, paths)
    output = {
        "min_lon": 0.0,
        "max_lon": 9.0,
        "min_lat": 0.0,
        "max_lat": 9.0,
        "min_time": "0",
        "max_time": "9",
    }
    assert kwargs_search == output

    kwargs_search = {
        "min_time": 1,
        "max_time": 2,
        "model_name": "path",
        "project_name": "test_project",
    }
    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search, paths)
    output = {
        "min_lon": 0.0,
        "max_lon": 9.0,
        "min_lat": 0.0,
        "max_lat": 9.0,
        "min_time": 1,
        "max_time": 2,
    }
    assert kwargs_search == output

    kwargs_search = {
        "min_lon": 1,
        "max_lon": 2,
        "min_lat": 1,
        "max_lat": 2,
        "model_name": "path",
        "project_name": "test_project",
    }
    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search, paths)
    output = {
        "min_lon": 1,
        "max_lon": 2,
        "min_lat": 1,
        "max_lat": 2,
        "min_time": "0",
        "max_time": "9",
    }
    assert kwargs_search == output

    kwargs_search = {
        "min_time": "1",
        "max_time": "2",
        "min_lon": "1",
        "max_lon": "2",
        "min_lat": "1",
        "max_lat": "2",
        "model_name": "path",
        "project_name": "test_project",
    }
    with pytest.raises(KeyError):
        kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search, paths)


def test_find_bbox():
    paths = omsa.paths.Paths(project_name="projectA", cache_dir=project_cache)
    lonkey, latkey, bbox, p1 = omsa.utils.find_bbox(ds, paths, mask=None)

    assert lonkey == "lon"
    assert latkey == "lat"
    assert bbox == [0.0, 0.0, 9.0, 9.0]
    assert isinstance(p1, shapely.geometry.polygon.Polygon)


def test_shift_longitudes():

    ds = xr.Dataset()
    ds["lon"] = (
        "lon",
        np.linspace(0, 360, 5)[:-1],
        {"units": "degrees_east", "standard_name": "longitude", "axis": "X"},
    )
    assert all(omsa.shift_longitudes(ds).cf["longitude"] == [-180.0, -90.0, 0.0, 90.0])

    ds = xr.Dataset()
    ds["lon"] = (
        "lon",
        np.linspace(-180, 180, 5)[:-1],
        {"units": "degrees_east", "standard_name": "longitude", "axis": "X"},
    )
    assert all(omsa.shift_longitudes(ds).cf["longitude"] == ds.cf["longitude"])


def test_vocab(project_cache):
    paths = omsa.paths.Paths(project_name="projectA", cache_dir=project_cache)
    v1 = omsa.utils.open_vocabs("general", paths)
    v2 = omsa.utils.open_vocabs(["general"], paths)
    v3 = omsa.utils.open_vocabs(
        project_cache / pathlib.PurePath("vocab/general"), paths
    )
    v4 = cf_pandas.Vocab(project_cache / pathlib.PurePath("vocab/general.json"))
    TestCase().assertDictEqual(v1.vocab, v2.vocab)
    TestCase().assertDictEqual(v1.vocab, v3.vocab)
    TestCase().assertDictEqual(v1.vocab, v4.vocab)


def test_vocab_labels(project_cache):
    paths = omsa.paths.Paths(project_name="projectA", cache_dir=project_cache)
    v1 = omsa.utils.open_vocab_labels("vocab_labels", paths)
    v2 = omsa.utils.open_vocab_labels(
        project_cache / pathlib.PurePath("vocab/vocab_labels"), paths
    )
    TestCase().assertDictEqual(v1, v2)
