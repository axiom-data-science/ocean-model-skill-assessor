"""
A package to fully run the comparison between data and model to assess model skill.
"""

from importlib.metadata import PackageNotFoundError, version

from ocean_model_skill_assessor.accessor import SkillAssessorAccessor

from .main import make_catalog, run
from .paths import CAT_PATH, LOG_PATH, PROJ_DIR, VOCAB_DIR, VOCAB_PATH
from .utils import shift_longitudes


try:
    __version__ = version("ocean-model-skill-assessor")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
