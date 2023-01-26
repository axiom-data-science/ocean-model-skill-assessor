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

# CIOFS

```{code-cell} ipython3
import ocean_model_skill_assessor as omsa
```

Set up model and data catalogs.

```{code-cell} ipython3
project_name = "ciofs_ncei"
# compare sea water temperature
key = "temp"

# model set up
loc = "https://www.ncei.noaa.gov/thredds/dodsC/model-ciofs-agg/Aggregated_CIOFS_Fields_Forecast_best.ncd"
model_name = "model"
kwargs_open = dict(drop_variables=["ocean_time"])

# data catalog set up
catalog_name = "erddap"
kwargs = dict(server="https://erddap.sensors.ioos.us/erddap", category_search=["standard_name", key])
kwargs_search = dict(min_time="2020-6-1", max_time="2020-6-5", max_lat=61.5, max_lon=-149,
                     min_lat=56.8, min_lon=-156)
```

```{code-cell} ipython3
# set up model catalog
cat_model = omsa.make_catalog(project_name=project_name,
                              catalog_type="local",
                              catalog_name=model_name,
                              kwargs=dict(filenames=loc, skip_entry_metadata=True),
                              kwargs_open=kwargs_open,
                              save_cat=True)
```

```{code-cell} ipython3
# set up data catalog
cat_data = omsa.make_catalog(project_name=project_name,
                             catalog_type="erddap",
                             catalog_name=catalog_name,
                             kwargs=kwargs,
                             save_cat=True,
                             kwargs_search=kwargs_search,
                             vocab="standard_names")
```

```{code-cell} ipython3
# Plot discovered data locations
omsa.plot.map.plot_cat_on_map(catalog=catalog_name, project_name=project_name)
```

The image shows a map of the Cook Inlet area with black dots with numbered labels showing data locations.

+++

Plot first 3 datasets in the data catalog.

```{code-cell} ipython3
omsa.run(project_name=project_name, catalogs=catalog_name, model_name=model_name,
         vocabs=["general","standard_names"], key_variable=key, ndatasets=3)
```

The first plot shows time series of temperature from data station "edu_ucsd_cdip_236" and nearby model output.

The second plot shows the Cook Inlet region on a map with a red outline of the numerical model boundary along with a black dot and number "0" showing the data location from which the data was taken.

```{code-cell} ipython3

```
