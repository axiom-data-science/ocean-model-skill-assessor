---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
---

```{code-cell}
import ocean_model_skill_assessor as omsa
import cf_pandas as cfp
```

# How to use `ocean-model-skill-assessor`

... as a Python package. Other notebooks describe its command line interface uses.

But, this is written in parallel to the [CLI demo](https://ocean-model-skill-assessor.readthedocs.io/en/latest/demo_cli.html), but will be more brief.

There are three steps to follow for a set of model-data validation, which is for one variable:
1. Make a catalog for your model output.
2. Make a catalog for your data.
3. Run the comparison.

These steps will save files into a user application directory cache, along with a log. A project directory can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`.


## Make model catalog

```{code-cell}
cat_model = omsa.make_catalog(project_name="demo_local_package", catalog_type="local", catalog_name="model",
                  kwargs=dict(filenames="https://www.ncei.noaa.gov/thredds/dodsC/model-ciofs-agg/Aggregated_CIOFS_Fields_Forecast_best.ncd",
                              skip_entry_metadata=True),
                  kwargs_open=dict(drop_variables="ocean_time"))
```

```{code-cell}
cat_model
```

## Make data catalog

Set up a catalog of the datasets with which you want to compare your model output. In this example, we use only known data file locations to create our catalog.

```{code-cell}
filenames = ["https://erddap.sensors.axds.co/erddap/tabledap/noaa_nos_co_ops_9455500.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature&time%3E=2022-01-01T00%3A00%3A00Z&time%3C=2022-01-06T00%3A00%3A00Z",
]

cat_data = omsa.make_catalog(project_name="demo_local_package", catalog_type="local", catalog_name="local",
                        kwargs=dict(filenames=filenames), kwargs_open=dict(blocksize=None))
```

```{code-cell}
cat_data
```

You may want to make a map of the data locations before doing your full run, especially in the case that you created a catalog from an ERDDAP server or similar. You can do this as follows:

```{code-cell}
omsa.plot.map.plot_cat_on_map(catalog=cat_data, project_name="demo_local_package")
```

The image shows a map around the dataset location in the Alaska region. The dataset location is marked with a black dot and marked with a numeric label.

+++

## Run comparison

Now that the model output and dataset catalogs are prepared, we can run the comparison of the two.

At this point we need to select a single variable to compare between the model and datasets, and this requires a little extra input. Because we don't know specifics about the format of any given input data file, variables will be interpreted with some flexibility in the form of a set of regular expressions. In the present case, we will compare the water temperature between the model and the datasets (the model output and datasets selected for our catalogs should contain the variable we want to compare). Several sets of regular expressions, called "vocabularies", are available with the package to be used for this purpose, and in this case we will use one called "general" which should match many commonly-used variable names. "general" is selected under `vocab_names`, and the particular key from the general vocabulary that we are comparing is selected with `key`.

See the vocabulary here.

```{code-cell}
cfp.Vocab(omsa.VOCAB_PATH("general"))
```

```{code-cell}
omsa.run(project_name="demo_local_package", catalogs=cat_data, model_name=cat_model,
         vocabs="general", key_variable="temp")
```

The plots show the time series comparisons for sea water temperatures of the model output and data at one location. Also shown is a map of the Cook Inlet region where the CIOFS model is located. An approximation of the numerical domain is shown along with the data location.
