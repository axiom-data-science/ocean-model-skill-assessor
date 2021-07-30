"""

"""

import pandas as pd
import ocean_model_skill_assessor


@pd.api.extensions.register_dataframe_accessor("omsa")
class SkillAssessorAccessor:
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
            raise TypeError('DataFrame index must be datetimes')

    @property
    def compute_stats(self):
        if not hasattr(self, '_compute_stats'):
            stats = ocean_model_skill_assessor.compute_stats(self.df['obs'], self.df['model'])
            self._compute_stats = stats
        return self._compute_stats

    def plot(self, title='title'):
        ocean_model_skill_assessor.time_series.plot(self.df['obs'], self.df['model'], title)
