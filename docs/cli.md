# Using OMSA through Command Line Interface (CLI)

Example commands are shown (but not run) below. You can copy these commands directly to a terminal window or command prompt.

This page is focused on explaining all the command line options, not demonstrating a workflow. For a more clear demonstration, check out the {doc}`Python package demo <demo>` or {doc}`CLI demo <demo_cli>`.

## Make catalog(s) for data and model

There are 3 types of catalogs in OMSA: local, erddap, and axds.

### Local catalog

Make a catalog with known local or remote file(s). Also use a local catalog to represent your model output.

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type local --catalog_name CATALOG_NAME --description "Catalog description" --kwargs filenames="[FILE1,FILE2]" --kwargs_open KWARG=VALUE --verbose --mode MODE

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `kwargs`: Some keyword arguments to make the local catalog. See `omsa.main.make_local_catalog()` for more details.
  * `filenames`: (Required) Where to find dataset(s) from which to make local catalog.
* `kwargs_open`: Keyword arguments to pass on to the appropriate intake open_* call for model or dataset.
* `verbose` Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log. Log is located in the project directory, which can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`. Default is True, to turn off use `--no-verbose`.
* `mode` mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.

#### Examples

##### Basic catalog for single dataset

    omsa make_catalog --project_name test1 --catalog_type local --catalog_name example_local_catalog --description "Example local catalog description" --kwargs filenames="[https://erddap.sensors.axds.co/erddap/tabledap/aoos_204.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature&time%3E=2022-01-01T00%3A00%3A00Z&time%3C=2022-01-06T00%3A00%3A00Z]"  --kwargs_open blocksize=None

##### Dataset with no lon/lat

When a dataset does not contain location information, you can input it as metadata to the catalog with the dataset. However, you need to input one filename and one set of metadata per catalog call.

Station page: https://tidesandcurrents.noaa.gov/stationhome.html?id=9455500

    omsa make_catalog --project_name test1 --catalog_type local --catalog_name example_local_catalog2 --kwargs filenames="[https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&application=NOS.COOPS.TAC.PHYSOCEAN&begin_date=20230109&end_date=20230109&station=9455500&time_zone=GMT&units=english&interval=6&format=csv]" --metadata minLongitude=-151.72 maxLongitude=-151.72 minLatitude=59.44 maxLatitude=59.44

##### Set up model

Use this approach to set up a catalog file for your model output, so that it can be used by OMSA. Use `skip_entry_metadata=True` when running for a model.

    omsa make_catalog --project_name test1 --catalog_type local --catalog_name model --kwargs filenames="https://www.ncei.noaa.gov/thredds/dodsC/model-ciofs-agg/Aggregated_CIOFS_Fields_Forecast_best.ncd" skip_entry_metadata=True  --kwargs_open drop_variables=ocean_time

### ERDDAP Catalog

Make a catalog from datasets available from an ERDDAP server using `intake-erddap`.

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type erddap --catalog_name CATALOG_NAME --description "Catalog description" --kwargs server=SERVER --kwargs_search min_lon=MIN_LON min_lat=MIN_LAT max_lon=MAX_LON max_lat=MAX_LAT min_time=MIN_TIME max_time=MAX_TIME search_for=SEARCH_TEXT --verbose --mode MODE

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `vocab_name`: Name of vocabulary to use from vocab dir. Options are "standard_names" and "general". See more information {doc}`here <add_vocab>`.
* `kwargs`: Some keyword arguments to make the ERDDAP catalog. See `intake-erddap.erddap_cat()` for more details.
  * `server`: ERDDAP server address, for example: "http://erddap.sensors.ioos.us/erddap"
  * `category_search`:
  * `use_source_constraints`: Any relevant search parameter defined in kwargs_search will be passed to the source objects as constraints.
  * `protocol`: str, default "tabledap"
  * `query_type`: Specifies how the catalog should apply the query parameters. Choices are ``"union"`` or ``"intersection"``. If the ``query_type`` is set to ``"intersection"``, then the set of results will be the intersection of each individual query made to ERDDAP. This is equivalent to a logical AND of the results. If the value is ``"union"`` then the results will be the union of each resulting dataset. This is equivalent to a logical OR.
  * other keyword arguments can be passed into the intake `Catalog` class
* `kwargs_search`: Keyword arguments to input to search on the server before making the catalog.
  * `min_lon`, `min_lat`, `max_lon`, `max_lat`: search for datasets within this spatial box
  * `min_time`, `max_time`: search for datasets with data within this time range
  * `model_name`: input a path to the model output to instead select the space and time search specifications based on the model. This input is specific to OMSA, not `intake-erddap`.
  * `search_for`: text-based search
* `verbose` Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log. Log is located in the project directory, which can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`. Default is True, to turn off use `--no-verbose`.
* `mode` mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.

#### Examples

##### Narrow search by input selections

Select a spatial box and time range over which to search catalog:

    omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name example_erddap_catalogA --description "Example ERDDAP catalog description" --kwargs server="https://erddap.sensors.ioos.us/erddap" --kwargs_search min_lon=-170 min_lat=53 max_lon=-165 max_lat=56 min_time=2022-1-1 max_time=2022-1-6

##### Narrow search with model output

Input model output to use to create the space search range, but choose time search range. We use the model catalog created in a previous example:

    omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name example_erddap_catalog --description "Example ERDDAP catalog description" --kwargs server="https://erddap.sensors.ioos.us/erddap" --kwargs_search model_name=model min_time=2022-1-1 max_time=2022-1-6

##### Narrow search also with `query_type`

You can additionally narrow your search by a text term by adding the `search_for` and `query_type` keyword inputs. This example searches for datasets containing the varaible "sea_surface_temperature" and, somewhere in the dataset metadata, the term "Timeseries". If we had wanted datasets that contain one OR the other, we could use `query_type=union`.

    omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name cat2 --kwargs server="https://erddap.sensors.ioos.us/erddap" standard_names="[sea_surface_temperature]" search_for="[Timeseries]" query_type=intersection

##### Variable selection by standard_name

Narrow your search by variable. For `intake-erddap` you can filter by the CF `standard_name` of the variable directly with the following.

    omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name cat1 --kwargs server="https://erddap.sensors.ioos.us/erddap" standard_names="[sea_surface_temperature,sea_water_temperature]"

##### Variable selection by pattern matching with vocab

You can return equivalent results in your catalog by searching with a variable nickname (the keys in the dictionary) along with a dictionary defining a vocabulary of regular expressions for matching what "counts" as a particular variable. To save a custom vocabulary to a location for this command, use the `Vocab` class in `cf-pandas` ([docs](https://cf-pandas.readthedocs.io/en/latest/demo_vocab.html#save-to-file)). A premade set of vocabularies aimed at use by ocean modelers is also available to use by name; see them with command `omsa vocabs`.  See more information {doc}`here <add_vocab>`. Suggested uses:
* axds catalog: vocab_name standard_names
* erddap catalog, IOOS: vocab_name standard_names
* erddap catalog, Coastwatch: vocab_name standard_names
* local catalog: vocab_name general

This is more complicated than simply defining the desired standard_names as shown in the previous example. However, it becomes useful when using other data files or model output which might have different variable names but could be recognized with variable matching through the vocabulary.

The example below uses the pre-defined vocabulary "standard_names" since we are using the IOOS ERDDAP server which uses standard_names as one of its search categories, and will search for matching variables by standard_name and matching the variable nickname "temp". The "standard_names" vocabulary is shown here and includes the standard_names from the previous example (it includes others too but they aren't present on the server). The regular expressions are set up to match exactly those standard_names. This is why we return the same results from either approach.

```
vocab = cfp.Vocab(omsa.VOCAB_PATH("standard_names"))
```

    omsa make_catalog --project_name test1 --catalog_type erddap --catalog_name cat3 --kwargs server="https://erddap.sensors.ioos.us/erddap" category_search="[standard_name,temp]" --vocab_name standard_names

### Catalog for Axiom assets

Make a catalog of Axiom Data Science-stored assets using `intake-axds`.

#### Available options

    omsa make_catalog --project_name PROJ_NAME --catalog_type axds --catalog_name CATALOG_NAME --description "Catalog description" --kwargs datatype="platform2 standard_names="[STANDARD_NAME1,STANDARD_NAME2]" page_size=PAGE_SIZE verbose=BOOL --kwargs_search min_lon=MIN_LON min_lat=MIN_LAT max_lon=MAX_LON max_lat=MAX_LAT min_time=MIN_TIME max_time=MAX_TIME search_for=SEARCH_TEXT --verbose --mode MODE

* `project_name`: Will be used as the name of the directory where the catalog is saved. The directory is located in a user application cache directory, the address of which can be found for your setup with  `omsa proj_path --project_name PROJ_NAME`.
* `catalog_type`: Type of catalog to make. Options are "erddap", "axds", or "local".
* `catalog_name`: Name for catalog.
* `description`: Description for catalog.
* `metadata`: Metadata for catalog.
* `vocab_name`: Name of vocabulary to use from vocab dir. Options are "standard_names" and "general". See more information {doc}`here <add_vocab>`.
* `kwargs`: Keyword arguments to make the AXDS catalog. See `intake-axds.axds_cat()` for more details.
  * `datatype`: Which type of Axiom asset to search for? Currently only "platform2" works and that is the default.
  * `keys_to_match`: Name of keys to match with system-available variable parameterNames using criteria. To filter search by variables, either input keys_to_match and a vocabulary or input standard_names.
  * `standard_names`: Standard names to select from Axiom search parameterNames. To filter search by variables, either input keys_to_match and a vocabulary or input standard_names.
  * `page_size`: Number of results. Fewer is faster. Note that default is 10. Note that if you want to make sure you get all available datasets, you should input a large number like 50000.
  * `verbose`: Set to True for helpful information.
  * other keyword arguments can be passed into the intake `Catalog` class
* `kwargs_search`: Keyword arguments to input to search on the server before making the catalog.
  * `min_lon`, `min_lat`, `max_lon`, `max_lat`: search for datasets within this spatial box
  * `min_time`, `max_time`: search for datasets with data within this time range
  * `model_name`: input a path to the model output to instead select the space and time search specifications based on the model. This input is specific to OMSA, not `intake-axds`.
  * `search_for`: text-based search
* `verbose` Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log. Log is located in the project directory, which can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`. Default is True, to turn off use `--no-verbose`.
* `mode` mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.

#### Examples

Many of the options available for an Axiom catalog are the same as for an ERDDAP catalog, so we show only a few combined examples here.

##### Narrow search by input selections

Select a box and time range over which to search catalog along with standard_name selections, with `verbose=True`.

    omsa make_catalog --project_name test1 --catalog_type axds --catalog_name example_axds_catalog1 --description "Example AXDS catalog description" --kwargs page_size=50000 standard_names="[sea_water_temperature]" verbose=True  --kwargs_search min_lon=-170 min_lat=53 max_lon=-165 max_lat=56 min_time=2000-1-1 max_time=2002-1-1

##### Same but with vocab

As in the ERDDAP catalog example above, we can instead get the same results by inputting a vocabulary to use, in this case "standard_names" which will map to variable names in Axiom systems, along with the variable nickname from the vocabulary to find: "temp".

    omsa make_catalog --project_name test1 --catalog_type axds --catalog_name example_axds_catalog2 --description "Example AXDS catalog description" --vocab_name standard_names --kwargs page_size=50000 keys_to_match="[temp]" --kwargs_search min_lon=-170 min_lat=53 max_lon=-165 max_lat=56 min_time=2000-1-1 max_time=2002-1-1

## Run model-data comparison

Note that if any datasets have timezones attached, they are removed before comparison with the assumption that the model output and data are in the same time zone.

For the final step of comparing the model output and datasets, we have to select a variable to compare, which means we have to select one or more vocabularies and one variable key. You should input the vocabularies used to create your catalogs and maybe also the "general" vocabulary if you used any known datasets, not from a specific server, since it then doesn't follow the same convention as the other files.

The datasets need to all cover the same time periods.

### Available options

    omsa run --project_name test1 --catalogs CATALOG_NAME1 CATALOG_NAME2 --vocab_names VOCAB1 VOCAB2 --key KEY --model_path PATH_TO_MODEL_OUTPUT --ndatasets NDATASETS --verbose --mode MODE

* `project_name`: Subdirectory in cache dir to store files associated together.
* `catalog_names`: Catalog name(s). Datasets will be accessed from catalog entries.
* `vocab_names`: Criteria to use to map from variable to attributes describing the variable. This is to be used with a key representing what variable to search for. This input is for the name of one or more existing vocabularies which are stored in a user application cache.
* `key`: Key in vocab(s) representing variable to compare between model and datasets.
* `model_name`: name of the model catalog we previously created
* `ndatasets`: Max number of datasets from each input catalog to use.
* `verbose` Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log. Log is located in the project directory, which can be checked on the command line with `omsa proj_path --project_name PROJECT_NAME`. Default is True, to turn off use `--no-verbose`.
* `mode` mode for logging file. Default is to overwrite an existing logfile, but can be changed to other modes, e.g. "a" to instead append to an existing log file.

### Example

Run a model-data comparison for the first 3 datasets in each of the 3 catalogs that we created previously in this notebook. Use vocabularies `standard_names` and `general` for variable matching. Match on the temperature variable with variable nickname "temp".

This example doesn't fully work because the combination of datasets are at different time periods and of different types that don't make sense to compare with the model output. It is shown as a template.

    omsa run --project_name test1 --catalog_names example_local_catalog example_erddap_catalog example_axds_catalog1 --vocab_names standard_names general --key temp --model_name model --ndatasets 1

## Utilities

A few handy utilities.

### Check location of project

With this you can check all of the project-related files you've created.

    omsa proj_path --project_name test1

### Check available vocabularies

    omsa vocabs

### Get information about a vocabulary

Return the path to the vocab file and the nicknames of the variables in the file.

    omsa vocab_info --vocab_name general
