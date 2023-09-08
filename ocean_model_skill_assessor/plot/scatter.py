"""Scatter plot."""

import matplotlib.pyplot as plt
import pandas as pd

import ocean_model_skill_assessor as omsa


def plot(
    reference: pd.DataFrame,
    sample: pd.DataFrame,
    nsubplots: int = 3,
    along_transect_distance: bool = True,
):
    """Scatter plot."""

    if along_transect_distance:
        reference["distance [km]"] = omsa.utils.calculate_distance(
            reference.cf["longitude"], reference.cf["latitude"]
        )
        sample["distance [km]"] = omsa.utils.calculate_distance(
            sample.cf["longitude"], sample.cf["latitude"]
        )

    fig, axes = plt.subplots(1, nsubplots, figsize=(15, 5))

    # plot reference (data)
    reference.plot.scatter(ax=axes[0], label="observation")  # , cmap=)
    sample.plot.scatter(ax=axes[1], label="model")  # , cmap=)

    # plot difference
    (reference - sample).scatter(ax=axes[2], label="difference")
