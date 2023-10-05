"""
Create paths, and handle file locations and vocabulary files.
"""


import shutil
import warnings

from pathlib import Path

import appdirs
import pandas as pd


class Paths(object):
    """Object to manage paths"""

    def __init__(self, project_name=None, cache_dir=None):
        """Initialize Paths object to manage paths in project.

        Parameters
        ----------
        project_name : str
            Subdirectory in cache dir to store files associated together.
        cache_dir : _type_, optional
            Input an alternative cache_dir if you prefer, esp for testing, by default None
        """
        # if project_name is None:
        #     warnings.warn("only `VOCAB_DIR` and `VOCAB_PATH` are available without supplying 'project_name'.")

        if cache_dir is None:
            # set up cache directories for package to use
            # user application cache directory, appropriate to each OS
            # dirs = AppDirs("ocean-model-skill-assessor", "axiom-data-science")
            # cache_dir = Path(dirs.user_cache_dir)
            cache_dir = Path(
                appdirs.user_cache_dir(
                    appname="ocean-model-skill-assessor", appauthor="axiom-data-science"
                )
            )
        self.cache_dir = cache_dir
        self.project_name = project_name

    @property
    def VOCAB_DIR(self):
        """Where to store and find vocabularies. Come from an initial set."""
        loc = self.cache_dir / "vocab"
        loc.mkdir(parents=True, exist_ok=True)
        loc_initial = Path(__file__).parent / "vocab"  # NEED THIS TO BE THE BASE PATH

        # copy vocab files to vocab cache location
        [shutil.copy(vocab_path, loc) for vocab_path in loc_initial.glob("*.json")]

        return loc

    @property
    def PROJ_DIR(self):
        """Return path to project directory."""
        assert self.project_name is not None
        path = self.cache_dir / f"{self.project_name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def CAT_PATH(self, cat_name):
        """Return path to catalog."""
        assert self.project_name is not None
        path = (self.PROJ_DIR / cat_name).with_suffix(".yaml")
        return path

    def VOCAB_PATH(self, vocab_name):
        """Return path to vocab."""
        path = (self.VOCAB_DIR / vocab_name).with_suffix(".json")
        return path

    @property
    def LOG_PATH(self):
        """Return path to vocab."""
        assert self.project_name is not None
        path = (self.PROJ_DIR / f"omsa").with_suffix(".log")

        # # if I can figure out how to make distinct logs per run
        # now = str(pd.Timestamp.today().isoformat())
        # path = PROJ_DIR(project_name) / "logs"
        # path.mkdir(parents=True, exist_ok=True)
        # path = (path / f"omsa_{now}").with_suffix(".log")
        return path

    @property
    def ALPHA_PATH(self):
        """Return path to alphashape polygon."""
        assert self.project_name is not None
        path = (self.PROJ_DIR / "alphashape").with_suffix(".txt")
        return path

    def MASK_PATH(self, key_variable):
        """Return path to mask cache for key_variable."""
        assert self.project_name is not None
        path = (self.PROJ_DIR / f"mask_{key_variable}").with_suffix(".nc")
        return path

    @property
    def MODEL_CACHE_DIR(self):
        """Return path to model cache directory."""
        assert self.project_name is not None
        path = self.PROJ_DIR / "model_output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def PROCESSED_CACHE_DIR(self):
        """Return path to processed data-model directory."""
        assert self.project_name is not None
        path = self.PROJ_DIR / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def OUT_DIR(self):
        """Return path to output directory."""
        assert self.project_name is not None
        path = self.PROJ_DIR / "out"
        path.mkdir(parents=True, exist_ok=True)
        return path
