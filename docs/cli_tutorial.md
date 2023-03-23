# Another example of using the command line interface

In this tutorial, we will first go through the basic steps of running a model-data comparison, then demonstrate how to vary the selections in each step. More information is also available on {doc}`extended CLI commands <cli>`. Note that every step shown in this tutorial could instead be run directly in Python with the Python package. More information on that is available in the {doc}`Python package demo <demo>`.

## Initial example

### Make model catalog

We use a package called [Intake](https://intake.readthedocs.io/en/latest/) to make catalogs for the models and datasets we use because it allows us to put into the catalog itself all of the unique flags and processing necessary to open the files. In turn, this allows us to work with the catalogs to read datasets in a generic, programmatic, and easy way.

Our first step is to make such a catalog for the model output. Here is the command for that.


    omsa make_catalog --project_name demo_local_B \
                      --catalog_type local \
                      --catalog_name model \
                      --kwargs filenames="https://www.ncei.noaa.gov/thredds/dodsC/model-ciofs-agg/Aggregated_CIOFS_Fields_Forecast_best.ncd" skip_entry_metadata=True  \
                      --kwargs_open drop_variables=ocean_time


The inputs you should change are:
* `project_name` which we will use for all commands that are adding to the same model-data comparison so that files are put in the same location,
* `catalog_name` if you want to choose a different name for the resulting catalog file,
* `filenames` under `kwargs` which is where the link(s) to the model output goes,
* `kwargs_open` into which you put the keyword arguments necessary to open the output. For netcdf files or opendap links, these will be passed to `xarray`. For csv files, they will be passed to `pandas`.


After running the command, check out the catalog file that was made.


### Make data catalog

Our next step is to make a catalog for the datasets we want to compare with the model output. They can be specific file locations (remote or local) or we could perform a search. We just need to end up with one or more Intake catalogs describing the datasets we want to use.

    omsa make_catalog --project_name demo_local_B \
                      --catalog_type local \
                      --catalog_name local \
                      --kwargs filenames="[https://erddap.sensors.axds.co/erddap/tabledap/nerrs_kachdwq.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature%2Csea_water_practical_salinity&time%3E=2022-01-01T00%3A00%3A00Z&time%3C=2022-01-06T00%3A00%3A00Z,https://erddap.sensors.axds.co/erddap/tabledap/nerrs_kacsdwq.csvp?time%2Clatitude%2Clongitude%2Cz%2Csea_water_temperature%2Csea_water_practical_salinity&time%3E=2022-01-01T00%3A00%3A00Z&time%3C=2022-01-06T00%3A00%3A00Z]" \
                      --kwargs_open blocksize=None


The inputs to change are:
* `project_name`, as above
* `catalog_type` should be "local" if you are making your own catalog from specific filenames, but could be other known types like "erddap" or "axds".
* `catalog_name`, as above
* `kwargs` to use depend on `catalog_type`. More information is available in the API docs.
* `kwargs_open`, as above, the keywords for opening the datasets. If the keywords are not the same for the datasets, then multiple catalogs should be created so that they are.

After running the command, check out the catalog file that was made.


### Run comparison

Now that the catalogs are ready, we can run our model-data comparison.

    omsa run --project_name demo_local_B \
             --catalog_names local \
             --model_name model \
             --vocab_names general \
             --key temp

Inputs to change are:
* `project_name` should be where the previously-made catalogs are stored
* `catalog_names` should be one or more names of data catalogs present in the `project_name` location
* `model_name` should be the name of the model catalog
* `vocab_names` is for interpreting variable names in the model output and datasets; more on this later. Several are pre-defined and available, and one or more can be input here.
* `key` is the variable to compare between the model and datasets. It must be defined in `vocab_names`.

After running the command, look at the resulting files in the location stated. You'll find the map of the model domain with data locations identified, computed statistics, and the time series comparisons. You can also look at the log file.


## Variations to try

### Vocabularies and using a different variable

Vocabs are relationships to link a nickname for a variable, like "temp" or "salt" for "temperature" and "salinity", to regular expressions to match variable names, since model and dataset variables could have any variety of names. Several pre-defined vocabs come with OMSA. Their location can be shown with:

    omsa vocabs

and more information can be found about one called "general" with:

    omsa vocab_info --vocab_name general

Alternatively, you could just open the files themselves for inspection.

Let's use a different variable key than we used above for another model-data comparison — one that we know is available in the datasets and model output, `salt`:

    omsa run --project_name demo_local_B \
             --catalog_names local \
             --model_name model \
             --vocab_names general \
             --key salt

Look at output files.

### Use a package to search for data

Instead of having to know about certain datasets to use for our comparison, we could instead search for data to use using, for example, [`intake-erddap`](https://intake-erddap.readthedocs.io/).

    omsa make_catalog --project_name demo_local_B \
                      --catalog_type erddap \
                      --catalog_name erddap \
                      --vocab_name general \
                      --kwargs server="https://erddap.sensors.ioos.us/erddap" standard_names="[sea_water_temperature]" query_type=intersection search_for="[cdip]" \
                      --kwargs_search min_lon=-154 min_lat=57.5 max_lon=-151 max_lat=61 min_time=2022-01-01 max_time=2022-01-06

Then run your comparison against this catalog:

    omsa run --project_name demo_local_B \
             --catalog_names erddap \
             --model_name model \
             --vocab_names general \
             --key temp


## Look at API

Look at {doc}`API docs <api>` for more info on using the functions.
