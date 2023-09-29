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

    return float((model - obs).mean())


def compute_correlation_coefficient(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray
) -> float:
    """Given obs and model signals, return Pearson product-moment correlation coefficient"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    # can't send nan's in
    inds = obs.notnull().values * model.notnull().values
    inds = inds.squeeze()

    return float(np.corrcoef(np.array(obs)[inds], np.array(model)[inds])[0, 1])


def compute_index_of_agreement(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray
) -> float:
    """Given obs and model signals, return Index of Agreement (Willmott 1981)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    ref_mean = obs.mean()
    num = ((obs - model) ** 2).sum()
    denom_a = np.abs(np.array(model - ref_mean))
    denom_b = np.abs(np.array(obs - ref_mean))
    denom = ((denom_a + denom_b) ** 2).sum()
    # handle underfloat
    if denom < 1e-16:
        return 1
    return float(1 - num / denom)


def compute_mean_square_error(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, centered=False
) -> float:
    """Given obs and model signals, return mean squared error (MSE)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    error = obs - model
    if centered:
        error += float(-obs.mean() + model.mean())
    return float((error**2).mean())


def compute_murphy_skill_score(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, obs_model=None
) -> float:
    """Given obs and model signals, return Murphy Skill Score (Murphy 1988)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    if not obs_model:
        obs_model = obs.copy()
        # # Have default solution be to skip the climatology
        # obs_model[:] = 0

        # if a obs forecast is not available, use mean of the *original* observations
        obs_model[:] = obs.mean()

    # # jesse's
    # mse_model = compute_mean_square_error(obs, model, centered=False)

    # # fake the name for the column
    # obs.name = "model"
    # mse_obs_model = compute_mean_square_error(obs_model, obs, centered=False)

    # if mse_obs_model <= 0:
    #     return -1
    # return float(1 - mse_model / mse_obs_model)

    # 1-((obs - model)**2).sum()/(obs**2).sum()
    return float(1 - ((obs - model) ** 2).sum() / ((obs - obs_model) ** 2).sum())


def compute_root_mean_square_error(
    obs: Union[pd.Series, xr.DataArray], model: xr.DataArray, centered=False
) -> float:
    """Given obs and model signals, return Root Mean Square Error (RMSE)"""

    assert isinstance(obs, (pd.Series, xr.DataArray))
    assert isinstance(model, xr.DataArray)

    mse = compute_mean_square_error(obs, model, centered=centered)
    return float(np.sqrt(mse))


def compute_descriptive_statistics(model: xr.DataArray, ddof=0) -> list:
    """Given obs and model signals, return the standard deviation"""

    assert isinstance(model, xr.DataArray)

    return list(
        [
            float(model.max()),
            float(model.min()),
            float(model.mean()),
            float(model.std(ddof=ddof)),
        ]
    )


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
