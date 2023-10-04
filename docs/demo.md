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
import cf_pandas as cfp
import xroms
```

# How to use `ocean-model-skill-assessor`

... as a Python package. Other notebooks describe its command line interface uses.

But, this is written in parallel to the {doc}`CLI demo <demo_cli>`, but will be more brief.

There are three steps to follow for a set of model-data validation, which is for one variable:
1. Make a catalog for your model output.
2. Make a catalog for your data.
3. Run the comparison.

These steps will save files into a user application directory cache, along with a log. A project directory can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`.

```{code-cell} ipython3
project_name = "demo_local_package"
```

## Make model catalog

We're using example ROMS model output that is available through `xroms` for our model.

```{code-cell} ipython3
url = xroms.datasets.CLOVER.fetch("ROMS_example_full_grid.nc")
kwargs = {
    "filenames": [url],
    "skip_entry_metadata": True,
}
cat_model = omsa.main.make_catalog(
                        catalog_type="local",
                        project_name=project_name,
                        catalog_name="model",
                        kwargs=kwargs,
                        return_cat=True,
)
```

```{code-cell} ipython3
cat_model
```

## Make data catalog

Set up a catalog of the datasets with which you want to compare your model output. In this example, we use only known data file locations to create our catalog.

Note that we need to include the "featuretype" and "maptype" in the metadata for the data sources. More information can be found on these items in the docs.

```{code-cell} ipython3
filenames = ["https://erddap.sensors.axds.co/erddap/tabledap/gov_ornl_cdiac_coastalms_88w_30n.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature&time%3E=2009-11-19T012%3A00%3A00Z&time%3C=2009-11-19T16%3A00%3A00Z",]

cat_data = omsa.make_catalog(project_name="demo_local_package",
                             catalog_type="local",
                             catalog_name="local",
                             kwargs=dict(filenames=filenames),
                             metadata={"featuretype": "timeSeries", "maptype": "point"})
```

```{code-cell} ipython3
cat_data
```

## Run comparison

Now that the model output and dataset catalogs are prepared, we can run the comparison of the two.

At this point we need to select a single variable to compare between the model and datasets, and this requires a little extra input. Because we don't know specifics about the format of any given input data file, variables will be interpreted with some flexibility in the form of a set of regular expressions. In the present case, we will compare the water temperature between the model and the datasets (the model output and datasets selected for our catalogs should contain the variable we want to compare). Several sets of regular expressions, called "vocabularies", are available with the package to be used for this purpose, and in this case we will use one called "general" which should match many commonly-used variable names. "general" is selected under `vocab_names`, and the particular key from the general vocabulary that we are comparing is selected with `key`.

See the vocabulary here.

```{code-cell} ipython3
paths = omsa.paths.Paths()
cfp.Vocab(paths.VOCAB_PATH("general"))
```

Now we run the model-data comparison. Check the API docs for details about the keyword inputs. Also note that the data has filler numbers for this time period which is why the comparison is so far off.

```{code-cell} ipython3
omsa.run(project_name="demo_local_package", catalogs=cat_data, model_name=cat_model,
         vocabs="general", key_variable="temp", interpolate_horizontal=False,
         check_in_boundary=False, plot_map=True, dd=5, alpha=20)
```

The plots show the time series comparisons for sea water temperatures of the model output and data at one location. Also shown is a map of the Mississippi river delta region where the model is located. An approximation of the numerical domain is shown along with the data location. Note that the comparison is poor because the data is missing for this time period.
