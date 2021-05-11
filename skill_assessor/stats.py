from typing import Tuple, Union

import numpy as np
from pandas import DataFrame
from xarray import DataArray


def _align(
    obs: Union[DataFrame, DataArray],
    model: Union[DataFrame, DataArray]
) -> DataFrame:
    """Aligns obs and model signals in time and returns a combined DataFrame"""
    if not isinstance(obs, DataFrame):
        obs = DataFrame(obs.to_pandas())
    if not isinstance(model, DataFrame):
        model = DataFrame(model.to_pandas())

    obs.rename(columns={obs.columns[0]: 'obs'}, inplace=True)
    model.rename(columns={model.columns[0]: 'model'}, inplace=True)
    aligned = obs.join(model).interpolate()
    return aligned


def compute_bias(
    obs: DataFrame,
    model: DataFrame
) -> DataFrame:
    """Given obs and model signals return bias (or, MSD in some communities)."""
    aligned_signals = _align(obs, model)
    return (aligned_signals['model'] - aligned_signals['obs']).mean()


def compute_correlation_coefficient(
    obs: DataFrame,
    model: DataFrame
) -> DataFrame:
    """Given obs and model signals, return Pearson product-moment correlation coefficient"""
    aligned_signals = _align(obs, model)
    return np.corrcoef(aligned_signals['obs'], aligned_signals['model'])[0, 1]


def compute_index_of_agreement(
    obs: DataFrame,
    model: DataFrame
) -> DataFrame:
    """Given obs and model signals, return Index of Agreement (Willmott 1981)"""
    aligned_signals = _align(obs, model)

    ref_mean = aligned_signals['obs'].mean()
    num = ((aligned_signals['obs'] - aligned_signals['model'])**2).sum()
    denom_a = (aligned_signals['model'] - ref_mean).abs()
    denom_b = (aligned_signals['obs'] - ref_mean).abs()
    denom = ((denom_a + denom_b)**2).sum()
    # handle underfloat
    if denom < 1e-16:
        return 1
    return 1 - num/denom


def compute_mean_square_error(
    obs: DataFrame,
    model: DataFrame,
    centered=False
) -> DataFrame:
    """Given obs and model signals, return mean square error (MSE)"""
    aligned_signals = _align(obs, model)

    error = aligned_signals['obs'] - aligned_signals['model']
    if centered:
        error += -aligned_signals['obs'].mean() + aligned_signals['model'].mean()
    return (error**2).mean()


def compute_murphy_skill_score(
    obs: DataFrame,
    model: DataFrame,
    obs_model=None
) -> DataFrame:
    """Given obs and model signals, return Murphy Skill Score (Murphy 1988)"""
    # if a obs forecast is not available, use mean of the *original* observations
    if not obs_model:
        obs_model = obs.copy()
        obs_model[:] = obs.mean().values[0]
        obs_model.rename(columns={'obs': 'obs_model'})

    mse_model = compute_mean_square_error(obs, model, centered=False)
    mse_obs_model = compute_mean_square_error(obs_model, obs, centered=False)
    if mse_obs_model <= 0:
        return -1
    return 1 - mse_model / mse_obs_model


def compute_root_mean_square_error(
    obs: DataFrame,
    model: DataFrame,
    centered=False
) -> DataFrame:
    """Given obs and model signals, return Root Mean Square Error (RMSE)"""
    mse = compute_mean_square_error(obs, model, centered=centered)
    return np.sqrt(mse)


def compute_descriptive_statistics(
    model: DataFrame,
    ddof=0
) -> Tuple:
    """Given obs and model signals, return the standard deviation"""
    return (np.max(model), np.min(model), np.mean(model), np.std(model, ddof=ddof))


def compute_stats(
    obs: DataFrame,
    model: DataFrame
) -> dict:
    """Compute stats and return as DataFrame"""

    return {
        'bias': compute_bias(obs, model),
        'corr': compute_correlation_coefficient(obs, model),
        'ioa': compute_index_of_agreement(obs, model),
        'mse': compute_mean_square_error(obs, model),
        'mss': compute_murphy_skill_score(obs, model),
        'rmse': compute_root_mean_square_error(obs, model),
        'descriptive': compute_descriptive_statistics(model)
    }
