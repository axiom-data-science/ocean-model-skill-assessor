ocean-model-skill-assessor
==============================
[![Build Status](https://img.shields.io/github/actions/workflow/status/axiom-data-science/ocean-model-skill-assessor/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions/workflows/test.yaml)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/ocean-model-skill-assessor.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/ocean-model-skill-assessor)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/ocean-model-skill-assessor/latest.svg?style=for-the-badge)](https://ocean-model-skill-assessor.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/actions/workflow/status/axiom-data-science/ocean-model-skill-assessor/linting.yaml?branch=main&label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions/workflows/linting.yaml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ocean_model_skill_assessor.svg?style=for-the-badge)](https://anaconda.org/conda-forge/ocean_model_skill_assessor)



A package to fully run the comparison between data and model to assess model skill.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>

## Run Demo without Installation

Click the binder button to open up a demonstration notebook in your browser window

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/axiom-data-science/ocean-model-skill-assessor/HEAD?labpath=docs%2FDemo-AK.ipynb)


## Installation

### Set up fresh environment for this package

First, make sure you have [Anaconda or Miniconda installed](https://conda.io/projects/conda/en/latest/user-guide/install/download.html).

Then, clone this repository:
``` bash
$ git clone https://github.com/axiom-data-science/ocean-model-skill-assessor.git
```

In the `ocean_model_skill_assessor` directory, install a conda environment with convenient packages for working with this package (beyond the requirements):
``` bash
$ conda env create -f environment.yml
```

Note that installing the packages is faster if you first install `mamba` to your base Python and then use `mamba` in place of `conda`.

Activate your new Python environment to use it with
``` bash
$ conda activate ocean-model-skill-assessor
```

### Install into existing Python environment

Install the package plus its requirements from `conda-forge` with
``` bash
$ conda install -c conda-forge ocean_model_skill_assessor
```

Or you can git clone the repository and then pip install it locally into your existing Python environment:
For local package install, in the `ocean_model_skill_assessor` directory:
``` bash
$ pip install -e .
```

### Extra packages for development

To also develop this package, install additional packages with:
``` bash
$ conda install --file requirements-dev.txt
```

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```

## Run Demo

In your terminal window, activate your Python environment if you are using one, then type `jupyter lab` in the `ocean_model_skill_assessor` directory. This will open into your browser window. Navigate to `docs/Demo-AK.ipynb` or any of the other notebooks and double-click to open. Inside a notebook, push `shift-enter` to run individual cells, or the play button at the top to run all cells, or select commands under the `Run` menu.
