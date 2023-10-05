import os

from unittest import mock

import intake
import numpy as np
import pandas as pd
import pytest

import ocean_model_skill_assessor as omsa


@pytest.fixture(scope="session")
def project_cache(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cache")
    return directory


def test_make_catalog_local(project_cache):
    paths = omsa.paths.Paths(project_name="projectA", cache_dir=project_cache)
    catloc2 = paths.CAT_PATH("catAlocal")

    kwargs = {"filenames": "filename.csv", "skip_entry_metadata": True}
    cat1 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=True,
        cache_dir=project_cache,
    )
    assert os.path.exists(catloc2)
    assert list(cat1) == ["filename"]
    assert cat1.name == "catAlocal"
    assert cat1["filename"].urlpath == "filename.csv"
    assert cat1["filename"].describe()["driver"] == ["csv"]
    assert cat1.description == "test local description"

    cat2 = intake.open_catalog(catloc2)
    assert cat1["filename"].describe() == cat2["filename"].describe()

    kwargs = {"filenames": "filenamenc.nc", "skip_entry_metadata": True}
    cat3 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=False,
    )
    assert cat3["filenamenc"].urlpath == "filenamenc.nc"
    assert cat3["filenamenc"].describe()["driver"] == ["netcdf"]

    kwargs = {
        "filenames": ["filenamenc.nc", "filename.csv"],
        "skip_entry_metadata": True,
    }
    cat4 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=False,
    )
    assert sorted(list(cat4)) == ["filename", "filenamenc"]

    with pytest.raises(ValueError):
        omsa.make_catalog(
            catalog_type="local",
            project_name="projectA",
        )


@mock.patch("intake.source.csv.CSVSource.read")
def test_make_catalog_local_read(read):

    df = pd.DataFrame(
        data={"time": np.arange(9), "lon": np.arange(9), "lat": np.arange(9)}
    )
    read.return_value = df

    kwargs = {"filenames": "filename.csv", "skip_entry_metadata": False}
    cat = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        kwargs=kwargs,
        return_cat=True,
        save_cat=False,
        metadata={"featuretype": "timeSeries", "maptype": "point"},
    )
    assert cat["filename"].metadata["minLongitude"] == 0.0
    assert cat["filename"].metadata["maxLatitude"] == 8.0
    assert pd.Timestamp(cat["filename"].metadata["minTime"]) == pd.Timestamp(
        "1970-01-01 00:00:00"
    )
