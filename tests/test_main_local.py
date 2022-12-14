import os
import pathlib

from unittest import mock

import intake

import ocean_model_skill_assessor as omsa


@mock.patch("ocean_model_skill_assessor.CAT_PATH")
def test_make_catalog_local(mock_cat_path, tmpdir):

    catloc2 = tmpdir / "projectA" / "catAlocal.yaml"
    mock_cat_path.return_value = catloc2

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
    assert os.path.exists(catloc2)
    assert list(cat1) == ["source0"]
    assert cat1.name == "catAlocal"
    assert cat1["source0"].urlpath == "filename.csv"
    assert cat1["source0"].describe()["driver"] == ["csv"]
    assert cat1.description == "test local description"

    cat2 = intake.open_catalog(catloc2)
    assert cat1["source0"].describe() == cat2["source0"].describe()

    kwargs = {"filenames": pathlib.PurePath("filename.nc")}
    cat3 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=False,
    )
    assert cat3["source0"].urlpath == "filename.nc"
    assert cat3["source0"].describe()["driver"] == ["netcdf"]

    kwargs = {"filenames": ["filename.nc", "filename.csv"]}
    cat4 = omsa.make_catalog(
        catalog_type="local",
        project_name="projectA",
        catalog_name="catAlocal",
        description="test local description",
        kwargs=kwargs,
        return_cat=True,
        save_cat=False,
    )
    assert sorted(list(cat4)) == ["source0", "source1"]
