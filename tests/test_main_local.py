import os
import pathlib

from unittest import mock

import intake

import ocean_model_skill_assessor as omsa


def test_make_catalog_local():

    kwargs = {"filenames": "filename.csv"}
    cat1 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=True,
    )
    assert os.path.exists(omsa.CAT_PATH("catAlocal", "projectA"))
    assert list(cat1) == ["source0"]
    assert cat1["source0"].urlpath == "filename.csv"
    assert cat1["source0"].describe()["driver"] == ["csv"]

    cat2 = intake.open_catalog(omsa.CAT_PATH("catAlocal", "projectA"))
    assert cat1["source0"].describe() == cat2["source0"].describe()

    kwargs = {"filenames": pathlib.PurePath("filename.nc")}
    cat1 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=True,
    )
    assert cat1["source0"].urlpath == "filename.nc"
    assert cat1["source0"].describe()["driver"] == ["netcdf"]

    kwargs = {"filenames": ["filename.nc", "filename.csv"]}
    cat1 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=True,
    )
    assert sorted(list(cat1)) == ["source0", "source1"]
