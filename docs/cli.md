# Using OMSA through Command Line Interface


## 0. Utilities

### Check location of project

    python CLI.py proj_path --project_name test1

Which returns something like `/Users/kthyng/Library/Caches/ocean-model-skill-assessor/test1`. Once you have that, you can check all of the project-related files you've created.

### Check available vocabularies

    python CLI.py vocabs

## 1. Make one or more catalogs

### ERDDAP Catalog

Make a catalog in subdirectory project directory "test2" of the datasets available from server https://erddap.sensors.ioos.us/erddap, and any resulting data should be read into xarray Datasets. To read into pandas `DataFrames` instead, substitute "dataframe" for "xarray".

    python CLI.py make_catalog --project_name test1 --erddap_server https://erddap.sensors.ioos.us/erddap --container xarray


### Catalog for Axiom assets

Make a catalog in subdirectory project directory "test1" of the first 12 platform datasets. Note that if you want to make sure you get all available datasets, you should input a large number like 50000.

    python CLI.py make_catalog --project_name test1 --catalog_type axds --kwargs page_size=12

Make a catalog of the first 10 platform datasets located in the spatial box and with data during the requested time range, and with `verbose=True`:

    python CLI.py make_catalog --project_name test1 --catalog_type axds --kwargs verbose=True --bbox -180 50 -158 66 --time_range 2022-1-1 2022-1-5

Input catalog name to be used in both file name and within the catalog:

    python CLI.py make_catalog --project_name test1 --catalog_type axds --catalog_name axds_test_cat

Input catalog description to be used within the catalog:

    python CLI.py make_catalog --project_name test1 --catalog_type axds --kwargs description="Description of this catalog."

Filter returned datasets for variables by standard_names. Note the syntax for inputting a list of standard_names through the command line interface.

    python CLI.py make_catalog --project_name test1 --catalog_type axds --kwargs standard_names='[sea_water_practical_salinity,sea_water_temperature]'

Alternatively, filter returned datasets for variables using the variable nicknames along with a vocabulary of regular expressions for matching what "counts" as a variable. To save a custom vocabulary to a location for this command, use the `Vocab` class in `cf-pandas` ([docs](https://cf-pandas.readthedocs.io/en/latest/demo_vocab.html#save-to-file)). A premade set of vocabularies is also available to use by name; see them with command `python CLI.py vocabs`. Suggested uses:
* axds catalog: vocab_name standard_names
* erddap catalog, IOOS: vocab_name erddap_ioos
* erddap catalog, Coastwatch: vocab_name erddap_coastwatch
* local catalog: vocab_name general

    python CLI.py make_catalog --project_name test1 --catalog_type axds --vocab_name standard_names --kwargs keys_to_match="[temp,salt]"
