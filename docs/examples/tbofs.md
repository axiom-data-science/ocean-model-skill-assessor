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

# TBOFS

Sea water temperature comparison between the Tampa Bay NOAA OFS model and IOOS ERDDAP Datasets.

```{code-cell} ipython3
import ocean_model_skill_assessor as omsa
```

```{code-cell} ipython3
project_name = "tbofs"
key = "temp"
```

```{code-cell} ipython3
# Model set up
loc = "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/TBOFS/fmrc/Aggregated_7_day_TBOFS_Fields_Forecast_best.ncd"
model_name = "model"
kwargs_open = dict(drop_variables="ocean_time")
# can't use chunks or model output won't be read in

# Data catalog set up
catalog_name = "erddap"
kwargs = dict(server="https://erddap.sensors.ioos.us/erddap", category_search=["standard_name", key])
kwargs_search = dict(model_name=model_name)
```

```{code-cell} ipython3
# Make model catalog
cat_model = omsa.make_catalog(project_name=project_name,
                              catalog_type="local",
                              catalog_name=model_name,
                              kwargs=dict(filenames=loc, skip_entry_metadata=True),
                              kwargs_open=kwargs_open,
                              save_cat=True)
```

```{code-cell} ipython3
# make data catalog
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

Image shows a map around Tampa Bay with data locations indicated in black with dots and numeric labels.

```{code-cell} ipython3
omsa.run(project_name=project_name, catalogs=catalog_name, model_name=model_name,
         vocabs=["general","standard_names"], key_variable=key, kwargs_map={"alpha": 20}, ndatasets=2)
```

The first image shows a time series comparison for station "edu_usf_marine_comps_c10" of temperature values between the data and the model.

The second image shows a map of the Tampa Bay region with a red outline of the approximate boundary of the numerical model along with a black dot for the data location and the number "0" labeling it.

```{code-cell} ipython3

```
