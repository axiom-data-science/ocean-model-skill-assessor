"""
A package to fully run the comparison between data and model to assess model skill.
"""

import shutil

from pathlib import Path

from appdirs import AppDirs
from pkg_resources import DistributionNotFound, get_distribution

import ocean_model_skill_assessor.accessor  # noqa: F401

from .main import make_catalog, run
from .plot import map, time_series
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


try:
    __version__ = get_distribution("ocean-model-skill-assessor").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"


# set up cache directories for package to use
# user application cache directory, appropriate to each OS
dirs = AppDirs("ocean-model-skill-assessor", "axiom-data-science")
cache_dir = Path(dirs.user_cache_dir)
VOCAB_DIR = cache_dir / "vocab"
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_DIR_INIT = Path(__path__[0]) / "vocab"  # NEED THIS TO BE THE BASE PATH

# copy vocab files to vocab cache location
[shutil.copy(vocab_path, VOCAB_DIR) for vocab_path in VOCAB_DIR_INIT.glob("*.json")]


def PROJ_DIR(project_name):
    """Return path to project directory."""
    path = cache_dir / f"{project_name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def CAT_PATH(cat_name, project_name):
    """Return path to catalog."""
    path = (cache_dir / project_name / cat_name).with_suffix(".yaml")
    return path


def VOCAB_PATH(vocab_name):
    """Return path to vocab."""
    path = (VOCAB_DIR / vocab_name).with_suffix(".json")
    return path
