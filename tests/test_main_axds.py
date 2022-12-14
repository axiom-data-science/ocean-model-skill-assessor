import os

from unittest import mock

import intake

import ocean_model_skill_assessor as omsa


class FakeResponse(object):
    def __init__(self):
        pass

    def json(self):
        res = {
            "results": [
                {
                    "uuid": "test_platform_parquet",
                    "label": "test_label",
                    "description": "Test description.",
                    "type": "platform2",
                    "start_date_time": "2019-03-15T02:58:51.000Z",
                    "end_date_time": "2019-04-08T07:54:56.000Z",
                    "source": {
                        "meta": {
                            "attributes": {
                                "institution": "example institution",
                                "geospatial_bounds": "POLYGON ((-156.25421 20.29439, -160.6308 21.64507, -161.15813 21.90021, -163.60744 23.30368, -163.83879 23.67031, -163.92656 23.83893, -162.37264 55.991, -148.04915 22.40486, -156.25421 20.29439))",
                            },
                            "variables": {"lon": "lon", "time": "time"},
                        },
                        "files": {
                            "data.csv.gz": {"url": "fake.csv.gz"},
                            "data.viz.parquet": {"url": "fake.parquet"},
                        },
                    },
                },
                {
                    "uuid": "test_platform_csv",
                    "label": "test_label",
                    "description": "Test description.",
                    "type": "platform2",
                    "start_date_time": "2019-03-15T02:58:51.000Z",
                    "end_date_time": "2019-04-08T07:54:56.000Z",
                    "source": {
                        "meta": {
                            "attributes": {
                                "institution": "example institution",
                                "geospatial_bounds": "POLYGON ((-156.25421 -20.29439, -160.6308 -21.64507, -161.15813 -21.90021, -163.60744 -23.30368, -163.83879 -23.67031, -163.92656 -23.83893, -162.37264 -55.991, -148.04915 -22.40486, -156.25421 -20.29439))",
                            },
                            "variables": {"lon": "lon", "time": "time"},
                        },
                        "files": {
                            "data.csv.gz": {"url": "fake.csv.gz"},
                        },
                    },
                },
            ]
        }
        return res


@mock.patch("ocean_model_skill_assessor.CAT_PATH")
@mock.patch("requests.get")
def test_make_catalog_axds_platform2(mock_requests, mock_cat_path, tmpdir):

    mock_requests.side_effect = [FakeResponse()]
    catloc2 = tmpdir / "projectA" / "catalog.yaml"
    mock_cat_path.return_value = catloc2

    cat1 = omsa.make_catalog(
        catalog_type="axds",
        project_name="projectA",
        catalog_name="catA",
        description="description of catalog",
        kwargs_search={"min_time": "2022-1-1", "max_time": "2022-1-2"},
        return_cat=True,
        save_cat=True,
    )

    assert os.path.exists(catloc2)
    assert sorted(list(cat1)) == ["test_platform_csv", "test_platform_parquet"]
    assert cat1.name == "catA"
    assert cat1["test_platform_csv"].urlpath == "fake.csv.gz"
    assert cat1["test_platform_parquet"].describe()["driver"] == ["parquet"]
    assert cat1.description == "description of catalog"
    assert cat1.metadata["kwargs_search"]["min_time"] == "2022-1-1"
    assert cat1.metadata["kwargs_search"]["max_time"] == "2022-1-2"

    mock_requests.side_effect = [FakeResponse()]
    cat2 = intake.open_axds_cat()
    assert (
        cat1["test_platform_parquet"].describe()
        == cat2["test_platform_parquet"].describe()
    )

    cat3 = intake.open_catalog(catloc2)
    assert (
        cat3["test_platform_parquet"].describe()
        == cat2["test_platform_parquet"].describe()
    )
