"""
Statistics functions.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

from pandas import DataFrame
from xarray import DataArray


def _align(
    obs: Union[DataFrame, DataArray], model: Union[DataFrame, DataArray]
) -> DataFrame:
    """Aligns obs and model signals in time and returns a combined DataFrame

    Returns
    -------
    A DataFrame indexed by time with one column 'obs' and one column 'model' which are at the model times and which does
    not extend in time beyond either's original time range.

    Notes
    -----
    Takes the model times as the correct times to interpolate obs to.
    """

    # if obs or model is a dask DataArray, output will be loaded in at this point
    if isinstance(obs, DataArray):
        obs = DataFrame(obs.to_pandas())
    elif isinstance(obs, pd.Series):
        obs = DataFrame(obs)
    if isinstance(model, DataArray):
        model = DataFrame(model.to_pandas())

    obs.rename(columns={obs.columns[0]: "obs"}, inplace=True)
    model.rename(columns={model.columns[0]: "model"}, inplace=True)

    # don't extrapolate beyond either time range
    min_time = max(obs.index.min(), model.index.min())
    max_time = min(obs.index.max(), model.index.max())
    obs = obs[min_time:max_time]
    model = model[min_time:max_time]

    # accounting for known issue for interpolation after sampling if indices changes
    # https://github.com/pandas-dev/pandas/issues/14297
    # get combined index of model and obs to first interpolate then reindex obs to model
    # otherwise only nan's come through
    ind = model.index.union(obs.index)
    obs = obs.reindex(ind).interpolate(method="time", limit=1).reindex(model.index)
    aligned = pd.concat([obs, model], axis=1)

    # Couldn't get this to work for me:
    # TODO: try flipping order of obs and model
    # aligned = obs.join(model).interpolate()
    return aligned


def compute_bias(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals return bias (or, MSD in some communities)."""

    # check if aligned already
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]
    return (model - obs).mean()


def compute_correlation_coefficient(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals, return Pearson product-moment correlation coefficient"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]
    # can't send nan's in
    inds = obs.notnull() * model.notnull()
    return np.corrcoef(obs[inds], model[inds])[0, 1]


def compute_index_of_agreement(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals, return Index of Agreement (Willmott 1981)"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]

    ref_mean = obs.mean()
    num = ((obs - model) ** 2).sum()
    denom_a = (model - ref_mean).abs()
    denom_b = (obs - ref_mean).abs()
    denom = ((denom_a + denom_b) ** 2).sum()
    # handle underfloat
    if denom < 1e-16:
        return 1
    return 1 - num / denom


def compute_mean_square_error(
    obs: DataFrame, model: DataFrame, centered=False
) -> DataFrame:
    """Given obs and model signals, return mean square error (MSE)"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]

    error = obs - model
    if centered:
        error += -obs.mean() + model.mean()
    return (error**2).mean()


def compute_murphy_skill_score(
    obs: DataFrame, model: DataFrame, obs_model=None
) -> DataFrame:
    """Given obs and model signals, return Murphy Skill Score (Murphy 1988)"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]

    # if a obs forecast is not available, use mean of the *original* observations
    if not obs_model:
        # import pdb; pdb.set_trace()
        obs_model = obs.copy()
        obs_model[:] = obs.mean()
        # obs_model[:] = obs.mean().values[0]
        # obs_model.rename(columns={'obs': 'obs_model'})

    mse_model = compute_mean_square_error(obs, model, centered=False)
    mse_obs_model = compute_mean_square_error(obs_model, obs, centered=False)
    if mse_obs_model <= 0:
        return -1
    return 1 - mse_model / mse_obs_model


def compute_root_mean_square_error(
    obs: DataFrame, model: DataFrame, centered=False
) -> DataFrame:
    """Given obs and model signals, return Root Mean Square Error (RMSE)"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]

    mse = compute_mean_square_error(obs, model, centered=centered)
    return np.sqrt(mse)


def compute_descriptive_statistics(model: DataFrame, ddof=0) -> Tuple:
    """Given obs and model signals, return the standard deviation"""
    return (model.max(), model.min(), model.mean(), model.std(ddof=ddof))


def compute_stats(obs: DataFrame, model: DataFrame) -> dict:
    """Compute stats and return as DataFrame"""

    # check if aligned
    if (len(obs) != len(model)) or (obs.index != model.index).any():
        aligned_signals = _align(obs, model)
        obs = aligned_signals["obs"]
        model = aligned_signals["model"]

    return {
        "bias": compute_bias(obs, model),
        "corr": compute_correlation_coefficient(obs, model),
        "ioa": compute_index_of_agreement(obs, model),
        "mse": compute_mean_square_error(obs, model),
        "mss": compute_murphy_skill_score(obs, model),
        "rmse": compute_root_mean_square_error(obs, model),
        "descriptive": compute_descriptive_statistics(model),
    }
