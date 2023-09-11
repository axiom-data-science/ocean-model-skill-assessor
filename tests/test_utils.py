from unittest import mock

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


@mock.patch("intake_xarray.base.DataSourceMixin.to_dask")
@mock.patch("intake.open_catalog")
def test_kwargs_search_from_model(mock_open_cat, mock_to_dask):

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

    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search)
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
    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search)
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
    kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search)
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
        kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search)


def test_find_bbox():
    lonkey, latkey, bbox, p1 = omsa.utils.find_bbox(ds)

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
