"""
A package to fully run the comparison between data and model to assess model skill.
"""

from pkg_resources import DistributionNotFound, get_distribution

import ocean_model_skill_assessor.accessor  # noqa: F401
# import ocean_model_skill_assessor.CLI

from .main import run
from .plot import map, time_series  # noqa: F401
from .stats import (  # noqa: F401
    compute_bias,
    compute_correlation_coefficient,
    compute_descriptive_statistics,
    compute_index_of_agreement,
    compute_mean_square_error,
    compute_murphy_skill_score,
    compute_root_mean_square_error,
    compute_stats,
)
from .utils import set_criteria



try:
    __version__ = get_distribution("ocean_model_skill_assessor").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
