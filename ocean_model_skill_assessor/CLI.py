"""
Command Line Interface.
"""

import argparse
import ocean_model_skill_assessor as omsa
import ocean_data_gateway as odg
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("config_file", 
                    help="Give path to configuration file. Should be yaml.")

args = parser.parse_args()
config_file = args.config_file

with open(config_file, "r") as ymlfile:
    cfg = yaml.load(ymlfile)

approach = cfg['approach']
loc_model = cfg['loc_model']
axds = cfg['axds']
bbox = cfg['bbox']
criteria = cfg['criteria']
erddap = cfg['erddap']
figname_map = cfg['figname_map']
figname_data_prefix = cfg['figname_data_prefix']
local = cfg['local']
only_search = cfg['only_search']
only_searchplot = cfg['only_searchplot']
parallel = cfg['parallel']
readers = cfg['readers']
run_qc = cfg['run_qc']
skip_units = cfg['skip_units']
stations = cfg['stations']
time_range = cfg['time_range']
var_def = cfg['var_def']
variables = cfg['variables']
xarray_kwargs = cfg['xarray_kwargs']

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
                  approach=approach,
                  loc_model=loc_model,
                  axds=axds,
                  bbox=bbox,
                  criteria=criteria,
                  erddap=erddap,
                  figname_map=figname_map,
                  figname_data_prefix=figname_data_prefix,
                  local=local,
                  only_search=only_search,
                  only_searchplot=only_searchplot,
                  parallel=parallel,
                  readers=readers,
                  run_qc=run_qc,
                  skip_units=skip_units,
                  time_range=time_range,
                  stations=stations,
                  var_def=var_def,
                  variables=variables,
                  xarray_kwargs=xarray_kwargs,
)