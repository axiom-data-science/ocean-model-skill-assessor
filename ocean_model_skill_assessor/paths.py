"""
Create paths, and handle file locations and vocabulary files.
"""


import shutil

from pathlib import Path

# from appdirs import AppDirs
import appdirs


# set up cache directories for package to use
# user application cache directory, appropriate to each OS
# dirs = AppDirs("ocean-model-skill-assessor", "axiom-data-science")
# cache_dir = Path(dirs.user_cache_dir)
cache_dir = Path(
    appdirs.user_cache_dir(
        appname="ocean-model-skill-assessor", appauthor="axiom-data-science"
    )
)
VOCAB_DIR = cache_dir / "vocab"
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_DIR_INIT = Path(__file__).parent / "vocab"  # NEED THIS TO BE THE BASE PATH
# import pdb; pdb.set_trace()
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


def LOG_PATH(project_name):
    """Return path to vocab."""
    path = (PROJ_DIR(project_name) / "omsa").with_suffix(".log")
    return path
