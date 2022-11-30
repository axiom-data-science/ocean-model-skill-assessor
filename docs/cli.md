# Using OMSA through Command Line Interface


## 1. Make one or more catalogs

### ERDDAP Catalog

Make a catalog in subdirectory project directory "test2" of the datasets available from server https://erddap.sensors.ioos.us/erddap, and any resulting data should be read into xarray Datasets. To read into pandas `DataFrames` instead, substitute "dataframe" for "xarray".

    python CLI.py make_catalog test1 --erddap_server https://erddap.sensors.ioos.us/erddap --container xarray


### Catalog for Axiom assets

Make a catalog in subdirectory project directory "test1" of the first 12 platform datasets.

    python CLI.py make_catalog test1 --axds_type platform2 --page_size 12

Make a catalog in subdirectory project directory "test1" of the first 10 platform datasets, and resulting data should be read into xarray Datasets. To read into pandas `DataFrames` instead, substitute "dataframe" for "xarray".

    python CLI.py make_catalog test1 --axds_type platform2 --container xarray

Make a catalog of the first 10 platform datasets located in the spatial box and with data during the requested time range:

    python CLI.py make_catalog test1 --axds_type platform2 --bbox -180 50 -158 66 --time_range 2022-1-1 2022-1-5

DESCRIBE VOCAB

    python CLI.py make_catalog test1 --axds_type platform2 --vocab vocab --nickname nickname 

