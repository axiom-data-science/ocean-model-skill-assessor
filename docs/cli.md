
# Using OMSA through Command Line Interface

Example commands will be run below in which case they are prefixed with `!` to run as a shell command instead of as Python code. In the terminal window, you should remove the `!` before running the command.

+++

## Make one or more catalogs

### Local catalog

Make a catalog with known local or remote file(s).

+++

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type local --catalog_name CATALOG_NAME --description "Catalog description" --kwargs filenames="[FILE1,FILE2]"

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `kwargs`: Keyword arguments to make the local catalog. See `omsa.main.make_local_catalog()` for more details.
  * `filenames`: (Required) Where to find dataset(s) from which to make local catalog.























+++

#### Examples

```{code-cell} ipython3
!omsa make_catalog --project_name test1 --catalog_type local --catalog_name example_local_catalog --description "Example local catalog description" --kwargs filenames="[https://researchworkspace.com/files/8114311/ecofoci_2011CHAOZ_CTD_Nutrient_mb1101.csv]"
```

### ERDDAP Catalog

Make a catalog from datasets available from an ERDDAP server using `intake-erddap`.

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type erddap --catalog_name CATALOG_NAME --description "Catalog description" --kwargs server=SERVER --kwargs_search min_lon=MIN_LON min_lat=MIN_LAT max_lon=MAX_LON max_lat=MAX_LAT min_time=MIN_TIME max_time=MAX_TIME search_for=SEARCH_TEXT

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `kwargs`: Keyword arguments to make the ERDDAP catalog. See `intake-erddap.erddap_cat()` for more details.
  * `server`: ERDDAP server address, for example: "http://erddap.sensors.ioos.us/erddap"
  * `category_search`:
  * `erddap_client`:
  * `use_source_constraints`:
  * `protocol`:
  * `metadata`:
  * other keyword arguments can be passed into the intake `Catalog` class
* `kwargs_search`: Keyword arguments to input to search on the server before making the catalog.
  * `min_lon`, `min_lat`, `max_lon`, `max_lat`: search for datasets within this spatial box
  * `min_time`, `max_time`: search for datasets with data within this time range
  * `model_path`: input a path to the model output to instead select the space and time search specifications based on the model. This input is specific to OMSA, not `intake-erddap`.
  * `search_for`: text-based search

#### Examples

Select a box and time range over which to search catalog:

```{code-cell} ipython3
!omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name example_erddap_catalog --description "Example ERDDAP catalog description" --kwargs server=https://erddap.sensors.ioos.us/erddap --kwargs_search min_lon=-170 min_lat=53 max_lon=-165 max_lat=56 min_time=2022-1-1 max_time=2022-1-2
```

Input model output to use to create the space search range, but choose time search range:

```{code-cell} ipython3
!omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name example_erddap_catalog --description "Example ERDDAP catalog description" --kwargs server=https://erddap.sensors.ioos.us/erddap --kwargs_search model_path=https://thredds.cencoos.org/thredds/dodsC/CENCOOS_CA_ROMS_FCST.nc min_time=2022-1-1 max_time=2022-1-2
```

### Catalog for Axiom assets

Make a catalog of Axiom Data Science-stored assets using `intake-axds`.

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type axds --catalog_name CATALOG_NAME --description "Catalog description" --kwargs datatype="platform2 standard_names="[STANDARD_NAME1,STANDARD_NAME2]" page_size=PAGE_SIZE verbose=BOOL --kwargs_search min_lon=MIN_LON min_lat=MIN_LAT max_lon=MAX_LON max_lat=MAX_LAT min_time=MIN_TIME max_time=MAX_TIME search_for=SEARCH_TEXT

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `kwargs`: Keyword arguments to make the ERDDAP catalog. See `intake-erddap.erddap_cat()` for more details.
  * `datatype`: Which type of Axiom asset to search for? Currently only "platform2" works and that is the default.
  * `keys_to_match`: Name of keys to match with system-available variable parameterNames using criteria. To filter search by variables, either input keys_to_match and a vocabulary or input standard_names.
  * `standard_names`: Standard names to select from Axiom search parameterNames. To filter search by variables, either input keys_to_match and a vocabulary or input standard_names.
  * `page_size`: Number of results. Fewer is faster. Note that default is 10. Note that if you want to make sure you get all available datasets, you should input a large number like 50000.
  * `verbose`: Set to True for helpful information.
  * other keyword arguments can be passed into the intake `Catalog` class
* `kwargs_search`: Keyword arguments to input to search on the server before making the catalog.
  * `min_lon`, `min_lat`, `max_lon`, `max_lat`: search for datasets within this spatial box
  * `min_time`, `max_time`: search for datasets with data within this time range
  * `model_path`: input a path to the model output to instead select the space and time search specifications based on the model. This input is specific to OMSA, not `intake-axds`.
  * `search_for`: text-based search

#### Examples

Select a box and time range over which to search catalog:

```{code-cell} ipython3
!omsa make_catalog --project_name test1 --catalog_type axds --catalog_name example_axds_catalog --description "Example AXDS catalog description" --kwargs standard_names='[sea_water_practical_salinity,sea_water_temperature]' verbose=True  --kwargs_search min_lon=-170 min_lat=53 max_lon=-165 max_lat=56 min_time=2000-1-1 max_time=2002-1-1 search_for=Bering
```

Input model output to use to create the space search range, but choose time search range:

```{code-cell} ipython3
!omsa make_catalog --project_name test1 --catalog_type axds --catalog_name example_axds_catalog --description "Example AXDS catalog description" --kwargs standard_names='[sea_water_practical_salinity,sea_water_temperature]' verbose=True --kwargs_search model_path=https://thredds.cencoos.org/thredds/dodsC/CENCOOS_CA_ROMS_FCST.nc min_time=2022-1-1 max_time=2022-1-2
```

Alternatively, filter returned datasets for variables using the variable nicknames along with a vocabulary of regular expressions for matching what "counts" as a variable. To save a custom vocabulary to a location for this command, use the `Vocab` class in `cf-pandas` ([docs](https://cf-pandas.readthedocs.io/en/latest/demo_vocab.html#save-to-file)). A premade set of vocabularies is also available to use by name; see them with command `omsa vocabs`. Suggested uses:
* axds catalog: vocab_name standard_names
* erddap catalog, IOOS: vocab_name erddap_ioos
* erddap catalog, Coastwatch: vocab_name erddap_coastwatch
* local catalog: vocab_name general

```
omsa make_catalog --project_name test1 --catalog_type axds --vocab_name standard_names --kwargs keys_to_match="[temp,salt]"
```

+++

## Run model-data comparison

Note that if any datasets have timezones attached, they are removed before comparison with the assumption that the model output and data are in the same time zone.

#### Available options

    omsa run --project_name test1 --catalog_names CATALOG_NAME1 CATALOG_NAME2 --vocab_names VOCAB1 VOCAB2 --key KEY --model_path PATH_TO_MODEL_OUTPUT --ndatasets NDATASETS

* `project_name`: Subdirectory in cache dir to store files associated together.
* `catalog_names`: Catalog name(s). Datasets will be accessed from catalog entries.
* `vocab_names`: Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
* `key`: Key in vocab(s) representing variable to compare between model and datasets.
* `model_path`: Where to find model output. Must be readable by xarray.open_mfdataset() (will be converted to list if needed).
* `ndatasets`: Max number of datasets from each input catalog to use.

#### Examples

Run a model-data comparison for the first 3 datasets in each of the 3 catalogs that we created previously in this notebook. Use vocabularies `erddap_ioos` and `general` for variable matching. Match on the temperature variable.

```{code-cell} ipython3
!omsa run --project_name test1 --catalog_names example_local_catalog example_erddap_catalog example_axds_catalog --vocab_name erddap_ioos general --key temp --model_path https://thredds.cencoos.org/thredds/dodsC/CENCOOS_CA_ROMS_FCST.nc --ndatasets 3
```

## Utilities

### Check location of project

```{code-cell} ipython3
!omsa proj_path --project_name test1
```

Which returns something like `/Users/kthyng/Library/Caches/ocean-model-skill-assessor/test1`. Once you have that, you can check all of the project-related files you've created.

### Check available vocabularies

```{code-cell} ipython3
!omsa vocabs
```
