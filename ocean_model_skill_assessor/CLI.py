"""
Command Line Interface.
"""

import argparse
import ocean_model_skill_assessor as omsa
import ocean_data_gateway as odg
import ast

parser = argparse.ArgumentParser()
parser.add_argument("loc_model", 
                    help="Give local or nonlocal location of model output.")
parser.add_argument("approach", 
                    help="Can be either 'region' or 'stations'.")
parser.add_argument("--xarray_kwargs",
                    help="Keyword arguments to pass to xarray.",
                    type=ast.literal_eval)
parser.add_argument("--time_range",
                    help="[min_time, max_time] for desired time range of search where each are strings that can be interpreted with pandas `Timestamp`.]",
                    type=ast.literal_eval)
parser.add_argument("--parallel",
                    help="Boolean, whether to run in parallel.",
                    action="store_true")
parser.add_argument("--readers",
                    help="Can specify which of the available readers to use in your search. Options are odg.erddap, odg.axds, and odg.local. Default is to use all.",
                    type=ast.literal_eval)
parser.add_argument("--variables",
                    help="Variables to search for.",
                    type=ast.literal_eval)
parser.add_argument("--criteria",
                    help="Note that this will both be used in this package and passed on to odg.",
                    type=ast.literal_eval)
parser.add_argument("--var_def",
                    help="Variable definitions.",
                    type=ast.literal_eval)
parser.add_argument("--local",
                    help="Options for local reader.",
                    type=ast.literal_eval)
parser.add_argument("--erddap",
                    help="Options for ERDDAP reader.",
                    type=ast.literal_eval)
parser.add_argument("--only_search",
                    help="Boolean, whether to stop after search.",
                    action="store_true")
parser.add_argument("--skip_units",
                    help="Boolean, whether to not deal with units.",
                    action="store_true")
parser.add_argument("--only_searchplot",
                    help="Boolean, whether to stop after search and map plot.",
                    action="store_true")
parser.add_argument("--stations",
                    help="List of stations",
                    type=list)


args = parser.parse_args()

loc_model = args.loc_model
approach = args.approach
xarray_kwargs = args.xarray_kwargs
time_range = args.time_range
parallel = args.parallel
readers = args.readers
variables = args.variables
criteria = args.criteria
var_def = args.var_def
local = args.local
erddap = args.erddap
only_search = args.only_search
skip_units = args.skip_units
only_searchplot = args.only_searchplot
stations = args.stations

omsa.set_criteria(criteria)

if readers is not None:
    input_readers = readers
    readers = []
    if 'local' in input_readers:
        readers.append(odg.local)
    if 'erddap' in input_readers:
        readers.append(odg.erddap)
    if 'axds' in input_readers:
        readers.append(odg.axds)

search = omsa.run(
                  loc_model=loc_model,
                  approach=approach,
                  xarray_kwargs=xarray_kwargs,
                  time_range=time_range,
                  parallel=parallel,
                  readers=readers,
                  variables=variables,
                  criteria=criteria,
                  var_def=var_def,
                  local=local,
                  erddap=erddap,
                  only_search=only_search,
                  skip_units=skip_units,
                  only_searchplot=only_searchplot,
                  stations=stations
)