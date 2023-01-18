"""
A package to fully run the comparison between data and model to assess model skill.
"""

from importlib.metadata import version, PackageNotFoundError
from ocean_model_skill_assessor.accessor import SkillAssessorAccessor

from .main import make_catalog, run
from .utils import shift_longitudes
from .paths import VOCAB_DIR, PROJ_DIR, CAT_PATH, VOCAB_PATH, LOG_PATH

try:
    __version__ = version("ocean-model-skill-assessor")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
