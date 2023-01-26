"""
Class to facilitate some functions directly on DataFrames.
"""

from pandas import DatetimeIndex
from pandas.api.extensions import register_dataframe_accessor

from ocean_model_skill_assessor.plot import time_series

from .stats import compute_stats


@register_dataframe_accessor("omsa")
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
        if not isinstance(df.index, DatetimeIndex):
            raise TypeError("DataFrame index must be datetimes")

    @property
    def compute_stats(self):
        """Run `compute_stats` on DataFrame."""
        if not hasattr(self, "_compute_stats"):
            stats = compute_stats(self.df["obs"], self.df["model"])
            self._compute_stats = stats
        return self._compute_stats

    def plot(self, **kwargs):
        """Plot."""
        time_series.plot(self.df["obs"], self.df["model"], **kwargs)
