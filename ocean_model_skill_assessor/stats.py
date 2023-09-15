"""
Statistics functions.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from pandas import DataFrame, Series, concat

from .paths import PROJ_DIR


# def check_aligned(obs: Union[DataFrame, xr.DataArray], model: Union[DataFrame, xr.DataArray],
# ):


def _align(
    obs: Union[DataFrame, xr.DataArray],
    model: Union[DataFrame, xr.DataArray],
    already_aligned: Optional[bool] = None,
) -> DataFrame:
    """Aligns obs and model signals in time and returns a combined DataFrame

    Parameters
    ----------
    already_aligned : optional, bool
        Way to override the alignment if user knows better. But still combines the obs and model together into one container.

    Returns
    -------
    A DataFrame indexed by time with one column 'obs' and one column 'model' which are at the model times and which does not extend in time beyond either's original time range.

    Notes
    -----
    Takes the obs times as the correct times to interpolate model to.
    """

    # guess about being already_aligned
    if already_aligned is None:
        if len(obs) == len(model):
            already_aligned = True
        else:
            already_aligned = False

    if already_aligned:
        if isinstance(obs, (Series, DataFrame)):
            obs.name = "obs"
            obs = DataFrame(obs)
            if isinstance(model, xr.DataArray):
                # if obs has multiindex, need to keep info for model too to match
                if isinstance(obs.index, pd.MultiIndex):
                    # need model to be dataset not dataarray to keep other coordinates
                    # when converting to dataframe
                    var_name = model.name
                    model = model.to_dataset()
                    indices = []
                    for index in ["T", "Z", "latitude", "longitude"]:
                        # if index in obs, have as index for model too
                        if index in obs.cf.keys():
                            # if index has only 1 unique value drop that index at this point
                            # for ilevel in and don't include for model indices

                            if (
                                len(
                                    obs.index.get_level_values(
                                        obs.cf[index].name
                                    ).unique()
                                )
                                > 1
                            ):
                                indices.append(model.cf[index].name)
                            else:
                                obs.index = obs.index.droplevel(obs.cf[index].name)
                    # Indices have to match exactly to concat correctly
                    # so if lon/lat are in indices, need to have interpolated to those values
                    # instead of finding nearest neighbors
                    model = model.to_pandas().reset_index().set_index(indices)[var_name]

                else:
                    model = model.squeeze().to_pandas()
                model.name = "model"
            elif isinstance(model, (Series, DataFrame)):
                model.name = "model"
                model = DataFrame(model)
            aligned = concat([obs, model], axis=1)
            aligned.index.names = obs.index.names
        else:  # both xarray
            obs.name = "obs"
            model.name = "model"

            aligned = xr.merge([obs, model])
        return aligned

    # if data is DataFrame/Series, bring model to pandas
    # either can be DataFrame or Series
    if isinstance(obs, (Series, DataFrame)):
        obs.name = "obs"
        # model.name = "model"
        obs = DataFrame(obs)
        if isinstance(model, xr.DataArray):
            # interpolate
            model = model.cf.interp(T=obs.cf["T"].unique())

            model = model.squeeze().to_pandas()
            model.name = "model"
            # after interpolating model to time, drop time index of obs if number of indices is 1
            # and there is more than one index
            if isinstance(obs.index, pd.core.indexes.multi.MultiIndex):
                unique_time_inds = (
                    obs.set_index([obs.cf["T"], obs.cf["Z"]])
                    .index.get_level_values(obs.cf["T"].name)
                    .unique()
                )
                if len(unique_time_inds) == 1:
                    obs = obs.droplevel(obs.cf["T"].name)
            # index_name = obs.index.name
            # obs.set_index([obs.cf["T"], obs.cf["Z"]]).cf["temp"].droplevel(obs.cf["T"].name)
            # if key_variable is not None:
            #     obs = obs.cf[key_variable]

            # # should be a DataFrame with 1 column
            # obs = obs.rename(columns={obs.columns[0]: "obs"})
            # model.name = "model"

        elif isinstance(model, xr.Dataset):
            raise TypeError(
                "Model output should be a DataArray, not Dataset, at this point."
            )
        elif isinstance(model, (Series, DataFrame)):
            model.name = "model"
            model = DataFrame(model)

        #     model = Series(model)
        # check if already aligned, in which case skip this
        if len(obs) == len(model):
            obs.name = "obs"
            model.name = "model"
            aligned = concat([obs, model], axis=1)
            aligned.index.name = obs.index.name
            return aligned
        else:
            # if obs or model is a dask DataArray, output will be loaded in at this point
            # if isinstance(obs, xr.DataArray):
            #     obs = DataFrame(obs.to_pandas())
            # elif isinstance(obs, Series):
            #     obs = DataFrame(obs)
            # obs = DataFrame(obs)

            # obs.rename(columns={obs.columns[0]: "obs"}, inplace=True)
            # model.rename(columns={model.columns[0]: "model"}, inplace=True)

            # don't extrapolate beyond either time range
            min_time = max(obs.index.min(), model.index.min())
            max_time = min(obs.index.max(), model.index.max())
            # try moving these later. They cause a problem when min_time==max_time
            # obs = obs[min_time:max_time]
            # model = model[min_time:max_time]

            # accounting for known issue for interpolation after sampling if indices changes
            # https://github.com/pandas-dev/pandas/issues/14297
            # get combined index of model and obs to first interpolate then reindex obs to model
            # otherwise only nan's come through
            ind = model.index.union(obs.index).unique()
            # only need to interpolate model if we are bringing model to match obs times
            model = (
                model.reindex(ind)
                .interpolate(method="time", limit=3)
                .reindex(obs.index.unique())
            )
            # obs = obs.reindex(ind).interpolate(method="time", limit=3).reindex(obs.index)
            obs = obs[min_time:max_time]
            model = model[min_time:max_time]
            # if use_index == "Z":
            #     model = model.T
            #     obs = obs.reset_index(drop=True).set_index(obs.cf["Z"].name)
            #     obs = obs.cf[key_variable]
            #     obs.name = "obs"
            #     model = model[model.columns[0]]
            #     model.name = "model"
            #     obs.index = np.negative(obs.index)
            #     aligned = concat([obs, model], axis=1)
            # else:
            # obs.name = "obs"
            # model.name = "model"
            aligned = concat([obs, model], axis=1)
            aligned.index.name = obs.index.name

        # Couldn't get this to work for me:
        # TODO: try flipping order of obs and model
        # aligned = obs.join(model).interpolate()

    # REMOVE THIS EVENTUALLY
    elif isinstance(obs, Series):
        # obs is pd.Series and model is xr.DataArray
        # need to change model to pd.Series
        if isinstance(model, xr.DataArray):
            model = Series(model.squeeze().to_pandas())
        elif isinstance(model, xr.Dataset):
            raise TypeError(
                "Model output should be a DataArray, not Dataset, at this point."
            )
        elif isinstance(model, DataFrame):
            model = Series(model)

        obs.name = "obs"
        model.name = "model"
        # obs is Series, model is Series now
        # check if already aligned, in which case skip this
        if len(obs) == len(model):
            return concat([obs, model], axis=1)
        else:
            # if obs or model is a dask DataArray, output will be loaded in at this point
            # if isinstance(obs, xr.DataArray):
            #     obs = DataFrame(obs.to_pandas())
            # elif isinstance(obs, Series):
            #     obs = DataFrame(obs)
            obs = DataFrame(obs)

            # obs.rename(columns={obs.columns[0]: "obs"}, inplace=True)
            # model.rename(columns={model.columns[0]: "model"}, inplace=True)

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
            obs = (
                obs.reindex(ind)
                .interpolate(method="time", limit=3)
                .reindex(model.index)
            )
            aligned = concat([obs, model], axis=1)
            aligned.index.name = obs.index.name

        # Couldn't get this to work for me:
        # TODO: try flipping order of obs and model
        # aligned = obs.join(model).interpolate()

    # otherwise have both be DataArrays
    else:

        # if already aligned, skip this
        if model.sizes == obs.sizes:
            return xr.merge([obs, model])

        # for all dimensions present, rename model to obs names so we
        # can merge the datasets
        for dim in ["T", "Z", "Y", "X"]:
            if dim in obs.cf.axes:
                key = obs.cf[dim].name
                model = model.rename({model.cf[dim].name: key})

        # don't extrapolate beyond either time range
        min_time = max(obs.cf["T"].min(), model.cf["T"].min())
        max_time = min(obs.cf["T"].max(), model.cf["T"].max())
        obs = obs.cf.sel(T=slice(min_time, max_time))
        model = model.cf.sel(T=slice(min_time, max_time))

        # some renaming
        obs.name = "obs"
        model.name = "model"

        # interpolate
        model = model.cf.interp(T=obs.cf["T"].values)

        aligned = xr.merge([obs, model])

    return aligned


def compute_bias(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals return bias."""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]
    return float((model - obs).mean())


def compute_correlation_coefficient(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals, return Pearson product-moment correlation coefficient"""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

    # can't send nan's in
    inds = obs.notnull() * model.notnull()
    inds = inds.squeeze()

    if isinstance(obs, Series):
        return float(np.corrcoef(obs[inds], model[inds])[0, 1])
    elif isinstance(obs, xr.DataArray):
        return float(
            np.corrcoef(obs.values[inds].squeeze(), model.values[inds].squeeze())[0, 1]
        )


def compute_index_of_agreement(obs: DataFrame, model: DataFrame) -> DataFrame:
    """Given obs and model signals, return Index of Agreement (Willmott 1981)"""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

    ref_mean = obs.mean()
    num = ((obs - model) ** 2).sum()
    if isinstance(obs, Series):
        denom_a = (model - ref_mean).abs()
        denom_b = (obs - ref_mean).abs()
    elif isinstance(obs, xr.DataArray):
        denom_a = xr.apply_ufunc(np.abs, (model - ref_mean))
        denom_b = xr.apply_ufunc(np.abs, (obs - ref_mean))
        # denom_a = (model - ref_mean).apply(np.fabs)
        # denom_b = (obs - ref_mean).apply(np.fabs)
    denom = ((denom_a + denom_b) ** 2).sum()
    # handle underfloat
    if denom < 1e-16:
        return 1
    return float(1 - num / denom)


def compute_mean_square_error(
    obs: DataFrame, model: DataFrame, centered=False
) -> DataFrame:
    """Given obs and model signals, return mean squared error (MSE)"""
    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

    error = obs - model
    if centered:
        error += -obs.mean() + model.mean()
    return float((error**2).mean())


def compute_murphy_skill_score(
    obs: DataFrame, model: DataFrame, obs_model=None
) -> DataFrame:
    """Given obs and model signals, return Murphy Skill Score (Murphy 1988)"""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

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
    obs: DataFrame, model: DataFrame, centered=False
) -> DataFrame:
    """Given obs and model signals, return Root Mean Square Error (RMSE)"""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

    mse = compute_mean_square_error(obs, model, centered=centered)
    return float(np.sqrt(mse))


def compute_descriptive_statistics(model: DataFrame, ddof=0) -> list:
    """Given obs and model signals, return the standard deviation"""
    return list(
        [
            float(model.max()),
            float(model.min()),
            float(model.mean()),
            float(model.std(ddof=ddof)),
        ]
    )


def compute_stats(obs: DataFrame, model: DataFrame) -> dict:
    """Compute stats and return as DataFrame"""

    # make sure aligned
    aligned_signals = _align(obs, model)
    obs, model = aligned_signals["obs"], aligned_signals["model"]

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
    source_name: str, stats: dict, project_name: str, key_variable: str, filename=None
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
        filename = PROJ_DIR(project_name) / f"stats_{source_name}_{key_variable}.yaml"

    with open(filename, "w") as outfile:
        yaml.dump(stats, outfile, default_flow_style=False)
