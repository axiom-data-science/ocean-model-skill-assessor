# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import pathlib
import sys

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# see https://pypi.org/project/setuptools-scm/ for details
from importlib.metadata import version as imversion


print("python exec:", sys.executable)
print("sys.path:", sys.path)
root = pathlib.Path(__file__).parent.parent.absolute()
os.environ["PYTHONPATH"] = str(root)
sys.path.insert(0, str(root))

import ocean_model_skill_assessor  # isort:skip

# -- Project information -----------------------------------------------------

project = "ocean-model-skill-assessor"
copyright = "2021-2023, Axiom Data Science"
author = "Axiom Data Science"

release = imversion("ocean-model-skill-assessor")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # "nbsphinx",
    "recommonmark",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.srclinks",
    "sphinx_markdown_tables",
    "myst_nb",
]

# for compiling notebooks with mystnb
# https://docs.readthedocs.io/en/stable/guides/jupyter.html#using-notebooks-in-other-formats
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    "_old_docs",
    ".ipynb",
    "notebooks",
]

html_extra_path = ["vocab_widget.html"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# furo variables
html_title = "ocean-model-skill-assessor documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- myst nb specific options ------

# https://myst-nb.readthedocs.io/en/v0.13.0/use/execute.html#execution-timeout
# had this message:
# WARNING: 'execution_timeout' is deprecated for 'nb_execution_timeout' [mystnb.config]
# WARNING: 'execution_allow_errors' is deprecated for 'nb_execution_allow_errors' [mystnb.config]
nb_execution_timeout = 600  # seconds.
nb_execution_allow_errors = False

# https://myst-nb.readthedocs.io/en/v0.9.0/use/execute.html
jupyter_execute_notebooks = "off"

# -- nbsphinx specific options ----------------------------------------------
# this allows notebooks to be run even if they produce errors.
# nbsphinx_allow_errors = True

# copied from cf-xarray
autosummary_generate = True

autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": True,
    "private-members": True,
}
napoleon_use_param = True
napoleon_use_rtype = True
