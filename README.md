ocean-model-skill-assessor
==============================
[![Build Status](https://img.shields.io/github/actions/workflow/status/axiom-data-science/ocean-model-skill-assessor/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions/workflows/test.yaml)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/ocean-model-skill-assessor.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/ocean-model-skill-assessor)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/ocean-model-skill-assessor/latest.svg?style=for-the-badge)](https://ocean-model-skill-assessor.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/actions/workflow/status/axiom-data-science/ocean-model-skill-assessor/linting.yaml?branch=main&label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions/workflows/linting.yaml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ocean-model-skill-assessor.svg?style=for-the-badge)](https://anaconda.org/conda-forge/ocean-model-skill-assessor)
[![Python Package Index](https://img.shields.io/pypi/v/ocean-model-skill-assessor.svg?style=for-the-badge)](https://pypi.org/project/ocean-model-skill-assessor)



A package to fully run the comparison between data and model to assess model skill.

You can run the analysis as a Python package or with a command-line interface.

There are three steps to follow for a set of model-data validation, which is for one variable:
1. Make a catalog for your model output.
2. Make a catalog for your data.
3. Run the comparison.

These steps will save files into a user application directory cache. See the demos for more details.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>


## Installation

### Set up environment

**NOTE**: Make sure you have [Anaconda or Miniconda installed](https://conda.io/projects/conda/en/latest/user-guide/install/download.html).

Create a conda environment called "omsa" that includes the package `ocean-model-skill-assessor`:
``` bash
$ conda create -n omsa -c conda-forge ocean-model-skill-assessor
```

Note that installing the packages is faster if you first install `mamba` to your base Python and then use "mamba" in place of all instances of "conda".

Activate your new Python environment to use it with
``` bash
$ conda activate omsa
```

Also install `cartopy` to be able to plot maps:
``` base
$ conda install -c conda-forge cartopy
```


### Install into existing environment

From `conda-forge`:
``` base
$ conda install -c conda-forge ocean-model-skill-assessor
```

From PyPI:
``` base
$ pip install ocean-model-skill-assessor
```

To plot a map of the model domain with data locations, you'll need to additionally install `cartopy`. If you used `conda` above:
``` base
$ conda install -c conda-forge cartopy
```

If you installed from PyPI, check out the instructions for installing `cartopy` [here](https://scitools.org.uk/cartopy/docs/latest/installing.html#building-from-source).


### Extra packages for development

To also develop this package, install additional packages with:
``` bash
$ conda install --file requirements-dev.txt
```

To then check code before committing and pushing it to github, locally run
``` bash
$ pre-commit run --all-files
```
