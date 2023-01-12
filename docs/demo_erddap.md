---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3.10.8 ('omsa')
  language: python
  name: python3
---

```{code-cell} ipython3
import ocean_model_skill_assessor as omsa
from IPython.display import Code, Image
```

# Demo of `ocean-model-skill-assessor` with `intake-erddap`

This demo runs command line interface (CLI) commands only, which is accomplished in a Jupyter notebook by prefacing commands with `!`. To transfer these commands to a terminal window, remove the `!` but otherwise keep commands the same.

More detailed docs about running with the CLI are [available](https://ocean-model-skill-assessor.readthedocs.io/en/latest/cli.html).

There are three steps to follow for a set of model-data validation, which is for one variable:
1. Make a catalog for your model output.
2. Make a catalog for your data.
3. Run the comparison.

These steps will save files into a user application directory cache.

## Make model catalog

Set up a catalog file for your model output. The user can input necessary keyword arguments – through `kwargs_open` – so that `xarray` will be able to read in the model output. Generally it is good to use `skip_entry_metadata` when using the `make_catalog` command for model output since we are using only one model and the entry metadata is aimed at being able to compare datasets.

In the following command, 
* `make_catalog` is the function being run from OMSA
* `demo_erddap` is the name of the project which will be used as the subdirectory name
* `local` is the type of catalog to choose when making a catalog for the model output regardless of where the model output is stored
* "model" is the catalog name which will be used for the file name and in the catalog itself
* Specific `kwargs` to be input to the catalog command are
  * `filenames` which is a string describing where the model output can be found. If the model output is available through a sequence of filenames instead of a single server address, represent them with a single `glob`-style statement, for example, "/filepath/filenameprefix_*.nc".
  * `skip_entry_metadata` use this when running `make_catalog` for model output
* `kwargs_open` all keywords required for `xr.open_dataset` or `xr.open_mfdataset` to successfully read your model output.

```{code-cell} ipython3
!omsa make_catalog --project_name demo_erddap --catalog_type local --catalog_name model --kwargs filenames=https://www.ncei.noaa.gov/thredds/dodsC/model-ciofs-agg/Aggregated_CIOFS_Fields_Forecast_best.ncd skip_entry_metadata=True  --kwargs_open drop_variables=ocean_time 
```

```{code-cell} ipython3
Code(filename=omsa.CAT_PATH("model", "demo_erddap"))
```

## Make data catalog 

Set up a catalog of the datasets with which you want to compare your model output. In this example, we search on an ERDDAP server to create our catalog.

In this step, we use the same `project_name` as in the previous step so as to put the resulting catalog file in the same subdirectory, we create a catalog of type "erddap" to search an ERDDAP server, we call this catalog file "erddap", we input the address for the ERDDAP server we're using, we search by "standard_name" on the server for variables that match with the variable nickname "temp" which can be found in the vocabulary "standard_names", the spatial extent to search is determined by checking the model in the model catalog, and a specific time range is chosen.

In the following command:
* `make_catalog` is the function being run from OMSA
* `demo_erddap` is the name of the project which will be used as the subdirectory name
* `erddap` is the type of catalog to choose when making a catalog from an ERDDAP server search
* "erddap" is the catalog name which will be used for the file name and in the catalog itself
* `vocab_name`: Name of the pre-defined vocabulary that provides regular expressions to match dataset variables to the nickname that they represent.
* Specific `kwargs` to be input to the catalog command are
  * `server` which is the location of the ERDDAP server.
  * `category_search`: "[standard_name,temp]" means to compare the variable nickname "temp" with the category of "standard_name" through the regular expressions defined in the vocabulary in `vocab_name`
* `kwargs_search` any keywords to narrow the search by, in this case:
  * `model_name` points to the model catalog we made in a previous step so that it can be read in and used to determine the spatial bounding box
  * we narrow the time range with `min_time` and `max_time`.
  * `search_for`: in this case we remove entries with the text "(HADS)" which are on land
  * search_for="-(HADS)" query_type="intersection"


    max_lat: 61.524627675
    max_lon: -148.92540748154363
    max_time: 2022-1-6
    min_lat: 56.7415606875
    min_lon: -156.48488313881575

```{code-cell} ipython3
!omsa make_catalog --project_name demo_erddap --catalog_type erddap --catalog_name erddap --vocab_name standard_names --kwargs server=https://erddap.sensors.ioos.us/erddap category_search="[standard_name,temp]" --kwargs_search min_lon=-156.5 min_lat=56.75 max_lat=61.5 max_lon=148.9 min_time=2022-1-1 max_time=2022-1-6
```

## Run comparison

Now that the model output and dataset catalogs are prepared, we can run the comparison of the two.

In this step, we use the same `project_name` as the other steps so as to keep all files in the same subdirectory. We input the data catalog name under `catalog_names` and the model catalog name under `model_name`. 

At this point we need to select a single variable to compare between the model and datasets, and this requires a little extra input. We do know the format of the data coming from the ERDDAP server because it is standardized, but the model output does not necessarily have variables tagged with standard_names to match. Accordingly, variables will be interpreted with some flexibility in the form of a set of regular expressions. In the present case, we will compare the water temperature between the model and the datasets (the model output and datasets selected for our catalogs should contain the variable we want to compare). Several sets of regular expressions, called "vocabularies", are available with the package to be used for this purpose, and in this case we will use one called "standard_names" (for standard names in the ERDDAP datasets) and "general" which should match many commonly-used variable names. "general" is selected under `vocab_names`, and the particular key from the general vocabulary that we are comparing is selected with `key`.

See the vocabulary here:

```{code-cell} ipython3
Code(filename=omsa.VOCAB_PATH("general"))
```

In the following command:
* `run` is the function being run from OMSA
* `demo_erddap` is the name of the project which will be used as the subdirectory name
* `catalog_names` are the names of any catalogs with datasets to include in the comparison. In this case we have just one called "erddap"
* `model_name` is the name of the model catalog we previously created
* `vocab_names` are the names of the vocabularies to use for interpreting which variable to compare from the model output and datasets. If multiple are input, they are combined together. The variable nicknames need to match in the vocabularies to be interpreted together.
* `key` is the nickname or alias of the variable as given in the input vocabulary
* `ndatasets` sets the max number of datasets per catalog to use in the comparison. Here we will just use 3.

Note that many of these datasets seem to be on land and therefore may not be able to be compared with the model output.

```{code-cell} ipython3
!omsa run --project_name demo_erddap --catalog_names erddap --model_name model --vocab_names general --key temp --ndatasets 3
```
