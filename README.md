ocean-model-skill-assessor
==============================
[![Build Status](https://img.shields.io/github/workflow/status/axiom-data-science/ocean-model-skill-assessor/Tests?logo=github&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/axiom-data-science/ocean-model-skill-assessor.svg?style=for-the-badge)](https://codecov.io/gh/axiom-data-science/ocean-model-skill-assessor)
[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/readthedocs/ocean-model-skill-assessor/latest.svg?style=for-the-badge)](https://ocean-model-skill-assessor.readthedocs.io/en/latest/?badge=latest)
[![Code Style Status](https://img.shields.io/github/workflow/status/axiom-data-science/ocean-model-skill-assessor/linting%20with%20pre-commit?label=Code%20Style&style=for-the-badge)](https://github.com/axiom-data-science/ocean-model-skill-assessor/actions)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ocean_model_skill_assessor.svg?style=for-the-badge)](https://anaconda.org/conda-forge/ocean_model_skill_assessor)



A package to fully run the comparison between data and model to assess model skill.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>

## Run Demo without Installation

Click the binder button to open up a demonstration notebook in your browser window

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/axiom-data-science/ocean-model-skill-assessor/HEAD?labpath=docs%2FDemo-AK.ipynb)


## Installation

### Set up fresh environment for this package

Clone the repo:
``` bash
$ git clone https://github.com/axiom-data-science/ocean-model-skill-assessor.git
```

In the `ocean_model_skill_assessor` directory, install conda environment:
``` bash
$ conda env create -f environment.yml
```

Note that this process is faster if you first install `mamba` to your base Python and then use `mamba` in place of `conda`.

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
