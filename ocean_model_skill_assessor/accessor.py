"""
Class to facilitate some functions directly on DataFrames.
"""

import pandas as pd
import xarray as xr

# from pandas import DatetimeIndex
from pandas.api.extensions import register_dataframe_accessor

from ocean_model_skill_assessor.plot import line, scatter, surface

from .stats import compute_stats


@register_dataframe_accessor("omsa")
@xr.register_dataset_accessor("omsa")
class SkillAssessorAccessor:
    """Class to facilitate some functions directly on DataFrames."""

    def __init__(self, dd):
        """

        don't validate with this because might not be time series

        Parameters
        ----------
        obj: DataFrame or Dataset.
            Should be observations and model, could be unloaded dask dataframe
        """
        # if isinstance(dd, pd.DataFrame):
        #     self._validate(dd)
        self.dd = dd

    @staticmethod
    def _validate(dd):
        """DataFrame must have datetimes as index."""
        if not isinstance(dd.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be datetimes")

    @property
    def compute_stats(self):
        """Run `compute_stats` on DataFrame."""
        if not hasattr(self, "_compute_stats"):
            stats = compute_stats(self.dd["obs"], self.dd["model"])
            self._compute_stats = stats
        return self._compute_stats

    def plot(self, featuretype=None, key_variable=None, **kwargs):
        """Plot."""
        import xcmocean

        # cmap and cmapdiff
        da = xr.DataArray(name=key_variable)

        # use featuretype to determine plot type, otherwise assume time series
        with xr.set_options(cmap_sequential=da.cmo.seq, cmap_divergent=da.cmo.div):
            if featuretype is not None:
                if featuretype == "timeSeries":
                    xname, yname, zname = (
                        self.dd.index.name or "index",
                        ["obs", "model"],
                        None,
                    )
                    xlabel, ylabel = "", key_variable
                    # xname, yname, zname = self.dd.cf["T"].name, ["obs","model"], None
                    line.plot(
                        self.dd.reset_index(),
                        xname,
                        yname,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        figsize=(15, 5),
                        **kwargs
                    )
                elif featuretype == "trajectoryProfile":
                    xname, yname, zname = "distance", "Z", ["obs", "model"]
                    surface.plot(
                        self.dd.reset_index(),
                        xname,
                        yname,
                        zname,
                        kind="scatter",
                        **kwargs
                    )
                    # surface.plot(xname, yname, self.dd["obs"], self.dd["model"], kind="scatter", **kwargs)
                    # scatter.plot(self.dd["obs"], self.dd["model"], **kwargs)
                elif featuretype == "timeSeriesProfile":
                    xname, yname, zname = "T", "Z", ["obs", "model"]
                    surface.plot(
                        self.dd.reset_index(),
                        xname,
                        yname,
                        zname,
                        kind="pcolormesh",
                        **kwargs
                    )
                    # surface.plot(xname, yname, self.dd["obs"].squeeze(), self.dd["model"].squeeze(), **kwargs)
                elif featuretype == "profile":
                    # use transpose so that index depth is plotted on y axis instead of x axis
                    xname, yname, zname = (
                        ["obs", "model"],
                        self.dd.index.name or "index",
                        None,
                    )
                    xlabel, ylabel = key_variable, yname
                    # xname, yname, zname = ["obs","model"], self.dd.cf["Z"].name, None
                    line.plot(
                        self.dd.reset_index(),
                        xname,
                        yname,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        figsize=(4, 8),
                        **kwargs
                    )
            else:
                xname, yname, zname = "index", ["obs", "model"], None
                line.plot(
                    self.dd.reset_index(), xname, yname, figsize=(15, 5), **kwargs
                )
                # time_series.plot(self.dd["obs"], self.dd["model"], **kwargs)


# @xr.register_dataset_accessor("omsa")
# class SkillAssessorAccessor:
#     """Class to facilitate some functions directly on Datasets."""

#     def __init__(self, ds):
#         """
#         Parameters
#         ----------
#         xarray_obj: Dataset
#             Should be observations and model, could be unloaded dask dataframe
#         """
#         self._validate(ds)
#         self.df = ds

#     # @staticmethod
#     # def _validate(df):
#     #     """DataFrame must have datetimes as index."""
#     #     if not isinstance(df.index, DatetimeIndex):
#     #         raise TypeError("DataFrame index must be datetimes")

#     @property
#     def compute_stats(self):
#         """Run `compute_stats` on Dataset"""
#         if not hasattr(self, "_compute_stats"):
#             stats = compute_stats(self.ds["obs"], self.ds["model"])
#             self._compute_stats = stats
#         return self._compute_stats

#     def plot(self, featuretype=None, **kwargs):
#         """Plot."""

#         # use featuretype to determine plot type, otherwise assume time series
#         if featuretype is not None:
#             if featuretype == "timeSeries":
#                 time_series.plot(self.df["obs"], self.df["model"], **kwargs)
#             elif featuretype == "trajectoryProfile":
#                 scatter.plot(self.df["obs"], self.df["model"], **kwargs)
#             elif featuretype == "timeSeriesProfile":
#                 surface.plot(self.df["obs"], self.df["model"], **kwargs)
#         else:
#             time_series.plot(self.df["obs"], self.df["model"], **kwargs)
