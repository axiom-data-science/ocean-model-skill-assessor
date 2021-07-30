from pkg_resources import DistributionNotFound, get_distribution

# extract_model is the package name (and subdirectory name)
# get_var_cf, select are function names
# from .model_assessor import get_var_cf, select

# extract_model is the package name
try:
    __version__ = get_distribution("model_assessor").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"
