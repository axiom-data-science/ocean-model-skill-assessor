import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ocean_model_skill_assessor as omsa

from ocean_model_skill_assessor.plot import line


def test_line():
    ref_times = pd.date_range(start="2000-12-30", end="2001-01-03", freq="6H")
    reference = pd.DataFrame(
        {"reference": np.sin(ref_times.values.astype("float32"))}, index=ref_times
    )

    sample_times = pd.date_range(start="2000-12-28", end="2001-01-04", freq="D")
    sample = pd.DataFrame(
        {"FAKE_SAMPLES": np.sin(sample_times.values.astype("float32"))},
        index=sample_times,
    )
    df = pd.concat([reference, sample])

    line.plot(df, xname="reference", yname="sample", title="test")


def test_map_no_cartopy():

    CARTOPY_AVAILABLE = omsa.plot.map.CARTOPY_AVAILABLE
    omsa.plot.map.CARTOPY_AVAILABLE = False

    maps = np.array(np.ones((2, 4)))
    figname = "test"
    dsm = xr.Dataset()

    with pytest.raises(ModuleNotFoundError):
        omsa.plot.map.plot_map(maps, figname, dsm)

    omsa.plot.map.CARTOPY_AVAILABLE = CARTOPY_AVAILABLE
