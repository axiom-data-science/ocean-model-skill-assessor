"""
Class to facilitate some functions directly on DataFrames.
"""

import re

import pandas as pd

import ocean_model_skill_assessor


regex = {
    "time": {"name": re.compile("\\bt\\b|(time|min|hour|day|week|month|year)[0-9]*")},
    "Z": {
        "name": re.compile(
            "(z|nav_lev|gdep|lv_|[o]*lev|bottom_top|sigma|h(ei)?ght|altitude|depth|"
            "isobaric|pres|isotherm)[a-z_]*[0-9]*"
        )
    },
    "Y": {"name": re.compile("y|j|nlat|nj")},
    "latitude": {"name": re.compile("y?(nav_lat|lat|gphi)[a-z0-9]*")},
    "X": {"name": re.compile("x|i|nlon|ni")},
    "longitude": {"name": re.compile("x?(nav_lon|lon|glam)[a-z0-9]*")},
}
regex["T"] = regex["time"]


@pd.api.extensions.register_dataframe_accessor("omsa")
class SkillAssessorAccessor:
    """Class to facilitate some functions directly on DataFrames."""

    def __init__(self, df):
        """
        Parameters
        ----------
        pandas_obj: DataFrame
            Should be observations, could be unloaded dask dataframe
        """
        self._validate(df)
        self.df = df

    @staticmethod
    def _validate(df):
        """DataFrame must have datetimes as index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be datetimes")

    @property
    def compute_stats(self):
        """Run `compute_stats` on DataFrame."""
        if not hasattr(self, "_compute_stats"):
            stats = ocean_model_skill_assessor.compute_stats(
                self.df["obs"], self.df["model"]
            )
            self._compute_stats = stats
        return self._compute_stats

    def plot(self, **kwargs):
        """Plot."""
        ocean_model_skill_assessor.plots.time_series.plot(
            self.df["obs"], self.df["model"], **kwargs
        )


@pd.api.extensions.register_dataframe_accessor("cf")
class cf_pandas:
    """Bring cf-xarray variable and coord/axis identification to pandas!"""

    def __init__(self, obj):
        """Initialize with pandas DataFrame."""
        self.df = obj

    def __getitem__(self, key):
        """This allows for df.cf[varname], etc."""

        # combine with regex from cf-xarray just for pandas cf
        criteria = ocean_model_skill_assessor.criteria
        criteria.update(regex)

        results = []
        if key in criteria:
            for criterion, patterns in criteria[key].items():
                results.extend(
                    list(
                        set([var for var in self.df.columns if re.match(patterns, var)])
                    )
                )

        key = list(set(results))[0]
        return self.df[key]
