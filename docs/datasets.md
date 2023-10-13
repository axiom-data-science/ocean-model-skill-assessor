# Catalog and dataset set up, NCEI feature type explainer

`ocean-model-skill-assessor` (OMSA) reads datasets from input `Intake` catalogs in order to abstract away the read in process. However, there are a few requirements of and suggestions for these catalogs, which are presented here.

## NCEI feature types

The NCEI netCDF feature types are useful because they describe what does and does not fit various definitions of oceanography data types. This defines types of dataset. More information is available [in general](https://www.ncei.noaa.gov/netcdf-templates) and for the current [NCEI NetCDF Templates 2.0](https://www.ncei.noaa.gov/data/oceans/ncei/formats/netcdf/v2.0/index.html). The following information may be useful for thinking about this and the necessary information below:

|                 | timeSeries     | profile        | timeSeriesProfile | trajectory (TODO)                     | trajectoryProfile     | grid (TODO)         |
|---              |---             |---             |---                |---                                    | ---                   | ---                 |
| Definition      | only t changes | only z changes | t and z change    | t, y, and x change                    | t, z, y, and x change | t changes, y/x grid |
| Data types      | mooring, buoy  | CTD profile    | moored ADCP       | flow through, 2D drifter | glider, transect of CTD profiles, towed ADCP, 3D drifter   | satellite, HF Radar |
| maptypes        | point  | point  | point  | point(s), line, box | point(s), line, box | box |
| X/Y are pairs (locstream) or grid | either locstream or grid | either locstream or grid | either locstream or grid | locstream | locstream | grid |
| Which dimensions are independent from X/Y choice? |
| T | Independent | Independent | Independent | match X/Y | match X/Y | Independent |
| Z | Independent | Independent | Independent | Independent | match X/Y | Independent |



## Requirements for datasets

### Requirements: pandas DataFrames

* `cf-pandas` must be able to identify a single column for each of the following keys:
  * T
  * Z
  * latitude
  * longitude

You can check a Catalog object with `omsa.utils.check_dataframe(df, no_Z)`.

Additionally, the variable you want to compare between model and data must be identifiable in both the dataset and model output using the custom vocabulary and a key in the vocabulary.


## Requirements and suggestions for Intake catalogs

### Requirements

* Metadata for a dataset must include:
  * an entry for "featuretype" that is a string of the NCEI-defined feature type that describes the dataset. Currently supported are `timeSeries`, `profile`, `trajectoryProfile`, `timeSeriesProfile` (`trajectory` and `grid` still to come).
  * an entry for "maptype" that is how to plot the dataset on a map. Currently supported are "point", "line", and "box".
  * "minLongitude", "maxLongitude", "minLatitude", "maxLatitude"
  * "minTime", "maxTime"

You can check a Catalog object with `omsa.utils.check_catalog(cat)`.


### Suggestions

* Do not encode indices for pandas DataFrames. If you do, though, they will be reset in OMSA.
* Note that DataFrames with a column that can be identified by `cf-pandas` as "T" will be parsed as datetimes.


## How to make an Intake catalog

* Use an Intake driver that supports direct catalog creation such as `intake-erddap`.
* Use `omsa.main.make_catalog()` or `omsa.main.make_local_catalog()`

## How to modify an Intake catalog

* coming soon, to add metadata to existing catalog
