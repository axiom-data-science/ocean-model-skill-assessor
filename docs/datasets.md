# Catalog and dataset set up

`ocean-model-skill-assessor` (OMSA) reads datasets from input `Intake` catalogs in order to abstract away the read in process. However, there are a few requirements of and suggestions for these catalogs, which are presented here.

## Requirements and suggestions for Intake catalogs

### Requirements

* Metadata for a dataset must include:
  * an entry for "featuretype" that is a string of the NCEI-defined feature type that describes the dataset. Currently supported are `timeSeries`, `profile`, `trajectoryProfile`, `timeSeriesProfile`.
  * an entry for "maptype" that is how to plot the dataset on a map. Currently supported are "point", "line", and "box".

### Suggestions

* Do not encode indices for pandas DataFrames. If you do, though, they will be reset in OMSA.
* Note that DataFrames with columns that can be identified by `cf-pandas` as containing datetimes will be parsed as such.


## How to make an Intake catalog

* Use an Intake driver that supports direct catalog creation such as `intake-erddap`.
* Use `omsa.main.make_catalog()` or `omsa.main.make_local_catalog()`

## How to modify an Intake catalog

* coming soon, to add metadata to existing catalog