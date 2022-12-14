from unittest import mock

import pandas as pd

import ocean_model_skill_assessor as omsa


SERVER_URL = "http://erddap.invalid/erddap"


@mock.patch("intake_erddap.erddap_cat.ERDDAPCatalog._load_metadata")
@mock.patch("pandas.read_csv")
def test_make_catalog_erddap(mock_read_csv, load_metadata_mock):
    load_metadata_mock.return_value = {}
    results = pd.DataFrame()
    results["datasetID"] = ["abc123"]
    mock_read_csv.return_value = results
    cat = omsa.make_catalog(
        catalog_type="erddap",
        project_name="test_project_erddap",
        catalog_name="test_catalog_erddap",
        description="description of test erddap catalog",
        kwargs={"server": SERVER_URL},
        kwargs_search={"min_time": "2022-1-1", "max_time": "2022-1-2"},
        return_cat=True,
        save_cat=False,
    )

    assert list(cat) == ["abc123"]
    assert cat.name == "test_catalog_erddap"
    assert cat["abc123"].describe()["args"]["server"] == SERVER_URL
    assert cat.description == "description of test erddap catalog"
    # assert cat.metadata["kwargs_search"]["min_time"] == '2022-1-1'
    # assert cat.metadata["kwargs_search"]["max_time"] == '2022-1-2'
