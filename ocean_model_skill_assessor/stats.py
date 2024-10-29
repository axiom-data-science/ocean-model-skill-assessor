"""
Statistics functions.
"""

from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .paths import Paths


def compute_bias(obs: Union[pd.Series, xr.DataArray], model: xr.DataArray) -> float:
    """Given obs and model signals return bias."""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     dim = "T"
    #     out = (model - obs).cf.mean(dim=dim)
    # else:
    out = float((model - obs).cf.mean())

    return out


def compute_correlation_coefficient(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray
) -> float:
    """Given obs and model signals, return Pearson product-moment correlation coefficient"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     # can't figure this one out and doesn't seem high priority
    #     out = np.nan
    # else:

    # can't send nan's in
    inds = obs.notnull().values * model.notnull().values
    inds = inds.squeeze()

    # out = float((model - obs).cf.mean())
    out = float(np.corrcoef(np.array(obs)[inds], np.array(model)[inds])[0, 1])

    return out


def compute_index_of_agreement(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray
) -> float:
    """Given obs and model signals, return Index of Agreement (Willmott 1981)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     dim = "T"
    #     ref_mean = obs.cf.mean(dim=dim)
    #     num = ((obs - model) ** 2).cf.sum(dim=dim)
    #     denom_a = np.abs(model - ref_mean)
    #     denom_b = np.abs(obs - ref_mean)
    #     # denom_a = np.abs(np.array(model - ref_mean))
    #     # denom_b = np.abs(np.array(obs - ref_mean))
    #     denom = ((denom_a + denom_b) ** 2).cf.sum(dim=dim)

    #     out = 1 - num / denom

    # else:
    ref_mean = obs.mean()
    num = ((obs - model) ** 2).sum()
    denom_a = np.abs(np.array(model - ref_mean))
    denom_b = np.abs(np.array(obs - ref_mean))
    denom = ((denom_a + denom_b) ** 2).sum()

    out = float(1 - num / denom)

    # handle underfloat
    if denom < 1e-16:
        return 1

    return out


def compute_mean_square_error(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, centered=False
) -> float:
    """Given obs and model signals, return mean squared error (MSE)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    error = obs - model

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     dim = "T"

    #     if centered:
    #         raise NotImplementedError("Centered not implemented for 3D")

    #     out = (error**2).cf.mean(dim=dim)

    # else:

    if centered:
        error += float(-obs.mean() + model.mean())
    out = float((error**2).mean())

    return out


def compute_murphy_skill_score(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, obs_model=None
) -> float:
    """Given obs and model signals, return Murphy Skill Score (Murphy 1988)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     dim = "T"

    #     if not obs_model:
    #         obs_model = obs.copy()
    #         # # Have default solution be to skip the climatology
    #         # obs_model[:] = 0

    #         # if a obs forecast is not available, use mean of the *original* observations
    #         obs_model[:] = obs.cf.mean(dim=dim)

    #     out = 1 - ((obs - model) ** 2).cf.sum(dim=dim) / ((obs - obs_model) ** 2).cf.sum(dim=dim)

    # else:

    if not obs_model:
        obs_model = obs.copy()
        # # Have default solution be to skip the climatology
        # obs_model[:] = 0

        # if a obs forecast is not available, use mean of the *original* observations
        obs_model[:] = obs.mean()

    out = float(1 - ((obs - model) ** 2).sum() / ((obs - obs_model) ** 2).sum())

    # # jesse's
    # mse_model = compute_mean_square_error(obs, model, centered=False)

    # # fake the name for the column
    # obs.name = "model"
    # mse_obs_model = compute_mean_square_error(obs_model, obs, centered=False)

    # if mse_obs_model <= 0:
    #     return -1
    # return float(1 - mse_model / mse_obs_model)

    # 1-((obs - model)**2).sum()/(obs**2).sum()
    return out


def compute_root_mean_square_error(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, centered=False
) -> float:
    """Given obs and model signals, return Root Mean Square Error (RMSE)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     mse = compute_mean_square_error(obs, model, centered=centered)
    #     out = np.sqrt(mse)

    # else:
    mse = compute_mean_square_error(obs, model, centered=centered)
    out = float(np.sqrt(mse))
    return out


def compute_descriptive_statistics(model: xr.DataArray, ddof=0) -> list:
    """Given obs and model signals, return the standard deviation"""

    assert isinstance(model, xr.DataArray)

    # # easier to consistently check model for this
    # # if 3D, assume we should calculate metrics over time dimension
    # if model.squeeze().ndim == 3:
    #     dim = "T"
    #     out = list(
    #     [
    #         model.cf.max(dim=dim),
    #         model.cf.min(dim=dim),
    #         model.cf.mean(dim=dim),
    #         model.cf.std(ddof=ddof, dim=dim),
    #     ]
    # )
    # else:
    out = list(
        [
            float(model.max()),
            float(model.min()),
            float(model.mean()),
            float(model.std(ddof=ddof)),
        ]
    )
    return out


def compute_stats(obs: Union[pd.Series, xr.DataArray], model: xr.DataArray) -> dict:
    """Compute stats and return as DataFrame"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    return {
        "bias": compute_bias(obs, model),
        "corr": compute_correlation_coefficient(obs, model),
        "ioa": compute_index_of_agreement(obs, model),
        "mse": compute_mean_square_error(obs, model),
        "ss": compute_murphy_skill_score(obs, model),
        "rmse": compute_root_mean_square_error(obs, model),
        "descriptive": compute_descriptive_statistics(model),
    }


def save_stats(
    source_name: str, stats: dict, key_variable: str, paths: Paths, filename=None
):
    """Save computed stats to file."""

    stats["bias"] = {
        "value": stats["bias"],
        "name": "Bias",
        "long_name": "Bias or MSD",
    }
    stats["corr"] = {
        "value": stats["corr"],
        "name": "Correlation Coefficient",
        "long_name": "Pearson product-moment correlation coefficient",
    }
    stats["ioa"] = {
        "value": stats["ioa"],
        "name": "Index of Agreement",
        "long_name": "Index of Agreement (Willmott 1981)",
    }
    stats["mse"] = {
        "value": stats["mse"],
        "name": "Mean Squared Error",
        "long_name": "Mean Squared Error (MSE)",
    }
    stats["ss"] = {
        "value": stats["ss"],
        "name": "Skill Score",
        "long_name": "Skill Score (Bogden 1996)",
    }
    stats["rmse"] = {
        "value": stats["rmse"],
        "name": "RMSE",
        "long_name": "Root Mean Square Error (RMSE)",
    }
    stats["descriptive"] = {
        "value": stats["descriptive"],
        "name": "Descriptive Statistics",
        "long_name": "Max, Min, Mean, Standard Deviation",
    }
    if "dist" in stats:
        stats["dist"] = {
            "value": stats["dist"],
            "name": "Distance",
            "long_name": "Distance in km from data location to selected model location",
        }

    if filename is None:
        filename = paths.PROJ_DIR / f"stats_{source_name}_{key_variable}.yaml"

    with open(filename, "w") as outfile:
        yaml.dump(stats, outfile, default_flow_style=False)
