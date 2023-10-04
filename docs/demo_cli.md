---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import ocean_model_skill_assessor as omsa
from IPython.display import Code, Image
```

# CLI demo of `ocean-model-skill-assessor` with known data files

This demo runs command line interface (CLI) commands only, which is accomplished in a Jupyter notebook by prefacing commands with `!`. To transfer these commands to a terminal window, remove the `!` but otherwise keep commands the same.

More detailed docs about running with the CLI are {doc}`available <cli>`.

There are three steps to follow for a set of model-data validation, which is for one variable:
1. Make a catalog for your model output.
2. Make a catalog for your data.
3. Run the comparison.

These steps will save files into a user application directory cache, along with a log. A project directory can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`.

## Make model catalog

Set up a catalog file for your model output. The user can input necessary keyword arguments – through `kwargs_open` – so that `xarray` will be able to read in the model output. Generally it is good to use `skip_entry_metadata` when using the `make_catalog` command for model output since we are using only one model and the entry metadata is aimed at being able to compare datasets.

In the following command,
* `make_catalog` is the function being run from OMSA
* `demo_local` is the name of the project which will be used as the subdirectory name
* `local` is the type of catalog to choose when making a catalog for the model output regardless of where the model output is stored
* "model" is the catalog name which will be used for the file name and in the catalog itself
* Specific `kwargs` to be input to the catalog command are
  * `filenames` which is a string describing where the model output can be found. If the model output is available through a sequence of filenames instead of a single server address, represent them with a single `glob`-style statement, for example, "/filepath/filenameprefix_*.nc".
  * `skip_entry_metadata` use this when running `make_catalog` for model output
* `kwargs_open` all keywords required for `xr.open_dataset` or `xr.open_mfdataset` to successfully read your model output.

```{code-cell} ipython3
# get local path for model output sample file from xroms
import xroms
url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
```

```{code-cell} ipython3
!omsa make_catalog --project_name demo_local --catalog_type local --catalog_name model --kwargs filenames=$url skip_entry_metadata=True
```

## Make data catalog

Set up a catalog of the datasets with which you want to compare your model output. In this example, we use only known data file locations to create our catalog.

In this step, we use the same `project_name` as in the previous step so as to put the resulting catalog file in the same subdirectory, we create a catalog of type "local" since we have known data locations, we call this catalog file "local", input the filenames as a list in quotes (this specific syntax is necessary for inputting a list in through the command line interface), and we input any keyword arguments necessary for reading the datasets.

In the following command:
* `make_catalog` is the function being run from OMSA
* `demo_local` is the name of the project which will be used as the subdirectory name
* `local` is the type of catalog to choose when making a catalog for the known data files
* "local" is the catalog name which will be used for the file name and in the catalog itself
* Specific `kwargs` to be input to the catalog command are
  * `filenames` which is a string or a list of strings pointing to where the data files can be found. If you are using a list, the syntax for the command line interface is `filenames="[file1,file2]"`.
* `kwargs_open` all keywords required for `xr.open_dataset` or `xr.open_mfdataset` or `pandas.open_csv`, or whatever method will ultimately be used to successfully read your model output. These must be applicable to all datasets represted by `filenames`. If they are not, run this command multiple times, one for each set of filenames and `kwargs_open` that match.

```{code-cell} ipython3
!omsa make_catalog --project_name demo_local --catalog_type local --catalog_name local --kwargs filenames="[https://erddap.sensors.axds.co/erddap/tabledap/gov_ornl_cdiac_coastalms_88w_30n.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature&time%3E=2009-11-19T012%3A00%3A00Z&time%3C=2009-11-19T16%3A00%3A00Z]" --metadata featuretype=timeSeries maptype=point
```

## Run comparison

Now that the model output and dataset catalogs are prepared, we can run the comparison of the two.

In this step, we use the same `project_name` as the other steps so as to keep all files in the same subdirectory. We input the data catalog name under `catalog_names` and the model catalog name under `model_name`.

At this point we need to select a single variable to compare between the model and datasets, and this requires a little extra input. Because we don't know anything about the format of any given input data file, variables will be interpreted with some flexibility in the form of a set of regular expressions. In the present case, we will compare the water temperature between the model and the datasets (the model output and datasets selected for our catalogs should contain the variable we want to compare). Several sets of regular expressions, called "vocabularies", are available with the package to be used for this purpose, and in this case we will use one called "general" which should match many commonly-used variable names. "general" is selected under `vocab_names`, and the particular key from the general vocabulary that we are comparing is selected with `key`.

See the vocabulary here.

```{code-cell} ipython3
import cf_pandas as cfp

paths = omsa.paths.Paths()
vocab = cfp.Vocab(paths.VOCAB_PATH("general"))
vocab
```

In the following command:
* `run` is the function being run from OMSA
* `demo_local` is the name of the project which will be used as the subdirectory name
* `catalog_names` are the names of any catalogs with datasets to include in the comparison. In this case we have just one called "local"
* `model_name` is the name of the model catalog we previously created
* `vocab_names` are the names of the vocabularies to use for interpreting which variable to compare from the model output and datasets. If multiple are input, they are combined together. The variable nicknames need to match in the vocabularies to be interpreted together.
* `key` is the nickname or alias of the variable as given in the input vocabulary

```{code-cell} ipython3
!omsa run --project_name demo_local --catalog_names local --model_name model --vocab_names general \
        --key temp \
      --kwargs_map label_with_station_name=True \
          --more_kwargs interpolate_horizontal=False check_in_boundary=False plot_map=True dd=5 alpha=20
```

## Look at results

Now we can look at the results from our comparison! You can find the location of the resultant files printed at the end of the `run` command output above. Or you can find the path to the project directory while in Python with:

```{code-cell} ipython3
paths = omsa.paths.Paths("demo_local")
paths.PROJ_DIR
```

Or you can use a command:

```{code-cell} ipython3
!omsa proj_path --project_name demo_local
```

Here we know the names of the files so show them inline.

First we see a map of the area around the Mississippi river delta, along with a red line outlining the approximate domain of the numerical model, and 1 black dot indicating 1 data location, marked with a the station name.

```{code-cell} ipython3
Image(paths.OUT_DIR / "map.png")
```

Here we see a time series comparison for station "gov_ornl_cdiac_coastalms_88w_30n". It shows in black the temperature values from the data and in red the comparable values from the model. The comparison time range is November 19, 2009 from 12 to 15:30. The lines are not similar because the data is actually missing during this time period. Statistical comparisons are also available in the title text.

```{code-cell} ipython3
Image(paths.OUT_DIR / "local_gov_ornl_cdiac_coastalms_88w_30n_temp.png")
```

```{code-cell} ipython3
import yaml
with open(paths.OUT_DIR / "local_gov_ornl_cdiac_coastalms_88w_30n_temp.yaml", "r") as stream:
    stats = yaml.safe_load(stream)
stats
```
