
import os
from unittest import mock
import intake
import ocean_model_skill_assessor as omsa


# string = """
# description: description
# metadata: {}
# name: Catalog
# sources:
#   source1:
#     args:
#       urlpath: file.nc
#     description: source description
#     driver: intake_xarray.netcdf.NetCDFSource
#     metadata: {}
#     name: source1
#     parameters: {}
# """

# @mock.patch("intake.open_axds_cat")
# @mock.patch("ocean_model_skill_assessor.CAT_PATH")
# def test_make_catalog_axds_platform2(mock_cat_path, mock_open_cat, tmpdir):
#     # make_catalog test1 --axds_type platform2 return cat
#     catloc = tmpdir / "cat.yaml"
#     with open(catloc, 'w') as f:
#         f.write(string)
#     cat_in = intake.open_catalog(catloc)
#     mock_open_cat.return_value = cat_in

#     catloc2 = tmpdir / "projectA" / "catA.yaml"
#     mock_cat_path.return_value = catloc2

#     cat = omsa.make_catalog(project_name="projectA", catalog_name="catA",
#                             axds_type="platform2", return_cat=True, save_cat=True)
#     assert cat == cat_in
#     assert os.path.exists(catloc2)
    
    
#     # make_catalog test1 --axds_type platform2 --container dataframe return cat
#     # make_catalog test1 --axds_type platform2 --container xarray return cat
#     # make_catalog test1 --axds_type platform2 --kwargs_search kw return cat
#     # make_catalog test1 --axds_type platform2 --vocab vocab --nickname nickname return cat
    

class FakeResponseSearch(object):
    def __init__(self):
        pass

    def json(self):
        res = {"results": [{"uuid": "test_uuid"}]}
        return res


class FakeResponseMeta(object):
    def __init__(self):
        pass

    def json(self):
        res = [
            {
                "data": {
                    "resources": {
                        "files": {
                            "data.csv.gz": {"url": "fake.csv.gz"},
                            "deployment.nc": {"url": "fake.nc"},
                        },
                    }
                }
            }
        ]

        return res


# @mock.patch("requests.get")
# def test_axds_catalog_platform_dataframe(mock_requests):
#     """Test basic catalog API: platform as dataframe."""

#     mock_requests.side_effect = [FakeResponseSearch(), FakeResponseMeta()]

#     cat = AXDSCatalog(datatype="platform2", outtype="dataframe")



@mock.patch("ocean_model_skill_assessor.CAT_PATH")
@mock.patch("requests.get")
def test_make_catalog_axds_platform2(mock_requests, mock_cat_path, tmpdir):

    mock_requests.side_effect = [FakeResponseSearch(), FakeResponseMeta()]
    catloc2 = tmpdir / "projectA" / "catalog.yaml"
    mock_cat_path.return_value = catloc2

    cat1 = omsa.make_catalog(project_name="projectA", catalog_name="catA", container="dataframe",
                            axds_type="platform2", return_cat=True, save_cat=True)

    assert os.path.exists(catloc2)

    mock_requests.side_effect = [FakeResponseSearch(), FakeResponseMeta()]
    cat2 = intake.open_axds_cat(datatype="platform2", outtype="dataframe")
    assert cat1["test_uuid"].describe() == cat2["test_uuid"].describe()

    cat3 = intake.open_catalog(catloc2)
    assert cat3["test_uuid"].describe() == cat2["test_uuid"].describe()
    