from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("ocean_model_skill_assessor").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

from .stats import (compute_bias,
                    compute_correlation_coefficient,
                    compute_index_of_agreement,
                    compute_mean_square_error,
                    compute_murphy_skill_score,
                    compute_root_mean_square_error,
                    compute_descriptive_statistics,
                    compute_stats)

from .plot import time_series

import ocean_model_skill_assessor.accessor
