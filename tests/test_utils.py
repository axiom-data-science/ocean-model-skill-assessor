from unittest import mock

import numpy as np
import pytest
import shapely.geometry
import xarray as xr

import ocean_model_skill_assessor as omsa


ds = xr.Dataset()
ds["time"] = (
    "time",
    np.arange(10),
    {"standard_name": "time"},
)
ds["lat"] = (
    "lat",
    np.arange(10),
    {"units": "degrees_north", "standard_name": "latitude"},
)
ds["lon"] = (
    "lon",
    np.arange(10),
    {"units": "degrees_east", "standard_name": "longitude"},
)


@mock.patch("xarray.open_mfdataset")
def test_kwargs_search_from_model(mock_xarray):

    kwargs_search = {"model_path": "path"}

    mock_xarray.return_value = ds

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

    kwargs_search = {"min_time": 1, "max_time": 2, "model_path": "path"}
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
        "model_path": "path",
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
        "model_path": "path",
    }
    with pytest.raises(KeyError):
        kwargs_search = omsa.utils.kwargs_search_from_model(kwargs_search)


def test_find_bbox():
    lonkey, latkey, bbox, p1 = omsa.utils.find_bbox(ds)

    assert lonkey == "lon"
    assert latkey == "lat"
    assert bbox == [0.0, 0.0, 9.0, 9.0]
    assert isinstance(p1, shapely.geometry.polygon.Polygon)
