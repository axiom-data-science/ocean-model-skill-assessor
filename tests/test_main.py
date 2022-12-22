


import pandas as pd
import pytest
import numpy as np
import xarray as xr
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
import cf_pandas as cfp
from unittest import mock
import ocean_model_skill_assessor as omsa


@mock.patch("intake.source.csv.CSVSource.read")
@mock.patch("xarray.open_mfdataset")
def test_run_variable(mock_ds, mock_read):
    """Test running with variable that is not present in catalog dataset."""
    
    # make catalog
    entries = {
        "test_source": LocalCatalogEntry(
            "test_source",
            description="",
            driver="csv",
            args={"urlpath": "fake.csv"},
            metadata={"minLongitude": 0,
                      "maxLongitude": 9,
                      "minLatitude": 0,
                      "maxLatitude": 9,
                      "minTime": "0",
                      "maxTime": "9"},
        )
    }
    # create catalog
    cat = Catalog.from_dict(
        entries,
        name="test_cat",
        description="",
        metadata={},
    )
    
    vocab = cfp.Vocab()
    vocab.make_entry("temp", "temp", attr="name")
    
    ds = xr.Dataset()
    ds["lon"] = (
        "lon",
        np.arange(9),
        {"units": "degrees_east", "standard_name": "longitude"},
    )    
    ds["temp"] = (
        "lon",
        np.arange(9),
        {"standard_name": "sea_water_temperature", 
         "coordinates": "lon"},
    )
    mock_ds.return_value = ds
    
    df = pd.DataFrame(
        columns=[
            "salt",
            "longitude",
            "latitude",
            "time",
        ]
    )
    mock_read.return_value = df

    with pytest.warns(RuntimeWarning):
        omsa.run(
        catalog_names=cat,
        project_name="projectB",
        key_variable="temp",
        model_path="fake.nc",
        vocabs=vocab,
        ndatasets=None,
        )
