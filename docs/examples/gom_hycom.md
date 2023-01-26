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

# Gulf of Mexico HYCOM

Compare sea water temperature between ERDDAP datasets and the model.

```{code-cell} ipython3
import ocean_model_skill_assessor as omsa
```

```{code-cell} ipython3
project_name = "gom_hycom"
key = "temp"
```

```{code-cell} ipython3
# Model set up information
loc = "http://tds.hycom.org/thredds/dodsC/GOMl0.04/expt_32.5/hrly"
model_name = "model"
kwargs_open = dict(drop_variables=["tau","time_run","surface_temperature_trend"], chunks="auto")

# ERDDAP data catalog set up information
catalog_name = "erddap"
kwargs = dict(server="https://erddap.sensors.ioos.us/erddap", category_search=["standard_name", key])
kwargs_search = dict(min_time="2019-2-1", max_time="2019-2-5",
                     min_lon=-98, max_lon=-96, min_lat=27, max_lat=30)
```

```{code-cell} ipython3
# create catalog for model
cat_model = omsa.make_catalog(project_name=project_name, 
                              catalog_type="local", 
                              catalog_name=model_name, 
                              kwargs=dict(filenames=loc, skip_entry_metadata=True),
                              kwargs_open=kwargs_open,
                              save_cat=True)
```

```{code-cell} ipython3
# create catalog for data
cat_data = omsa.make_catalog(project_name=project_name, 
                             catalog_type="erddap", 
                             catalog_name=catalog_name, 
                             kwargs=kwargs,
                             save_cat=True,
                             kwargs_search=kwargs_search,
                             vocab="standard_names")
```

```{code-cell} ipython3
# look at locations of all data found
omsa.plot.map.plot_cat_on_map(catalog=catalog_name, project_name=project_name)
```

The image shows a map of part of the Texas coastline. Overlaid are black dots, each numbered, to indicate a location of a dataset.

```{code-cell} ipython3
omsa.run(project_name=project_name, catalogs=catalog_name, model_name=model_name,
         vocabs=["general","standard_names"], key_variable=key, ndatasets=9)
```

A time series is shown comparing the temperatures in dataset "noaa_nos_co_ops_8773037" with nearby model output. The lines are reasonably similar.

Subsequently is shown a map of the Gulf of Mexico with a red outline of the approximate numerical domain and a single black dot with a number "0" showing the location of the dataset that was plotted.
