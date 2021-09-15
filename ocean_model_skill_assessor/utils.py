"""
Utility functions.
"""

import cf_xarray
import ocean_data_gateway as odg

import ocean_model_skill_assessor as omsa


def set_criteria(criteria):
    """Set up criteria."""

    if isinstance(criteria, str) and criteria[:4] == "http":
        criteria = odg.return_response(criteria)

    cf_xarray.set_options(custom_criteria=criteria)
    omsa.criteria = criteria
