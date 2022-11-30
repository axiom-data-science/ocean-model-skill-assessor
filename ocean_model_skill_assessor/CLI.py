"""
Command Line Interface.
"""

import argparse

# import ocean_data_gateway as odg
# import yaml

import ocean_model_skill_assessor as omsa


parser = argparse.ArgumentParser()

parser.add_argument("action", help="make_catalog")
parser.add_argument("project_name", help="All saved items will be stored in a subdirectory call `project_name` in the user application cache.")
parser.add_argument("--axds_type")
parser.add_argument("--erddap_server")
parser.add_argument("--nickname")
parser.add_argument("--bbox", nargs=4, help="min_lon min_lat max_lon max_lat")
parser.add_argument("--time_range", nargs=2, help="min_time max_time")
parser.add_argument("--container", help="dataframe or xarray")
parser.add_argument("--catalog_name")
parser.add_argument("--vocab_name")
parser.add_argument("--page_size", help="max number of datasets")

args = parser.parse_args()
# print(args)
# import pdb; pdb.set_trace()

if args.bbox is not None:
    kwargs_search = {"min_lon": args.bbox[0],
                            "min_lat": args.bbox[1],
                            "max_lon": args.bbox[2],
                            "max_lat": args.bbox[3],}
else:
    kwargs_search = None

omsa.make_catalog(project_name=args.project_name,
             catalog_name=args.catalog_name,
             nickname=args.nickname,
            #  filenames: Optional[Union[Sequence,str]] = None,
             erddap_server=args.erddap_server,
             axds_type=args.axds_type,
             kwargs_search=kwargs_search,
             vocab=args.vocab_name,
             page_size=args.page_size,
             container=args.container,
                 )


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "config_file", help="Give path to configuration file. Should be yaml."
# )

# args = parser.parse_args()
# config_file = args.config_file

# with open(config_file, "r") as ymlfile:
#     cfg = yaml.load(ymlfile)

# approach = cfg["approach"]
# loc_model = cfg["loc_model"]
# axds = cfg["axds"]
# bbox = cfg["bbox"]
# criteria = cfg["criteria"]
# erddap = cfg["erddap"]
# figname_map = cfg["figname_map"]
# figname_data_prefix = cfg["figname_data_prefix"]
# local = cfg["local"]
# only_search = cfg["only_search"]
# only_searchplot = cfg["only_searchplot"]
# parallel = cfg["parallel"]
# readers = cfg["readers"]
# run_qc = cfg["run_qc"]
# skip_units = cfg["skip_units"]
# stations = cfg["stations"]
# time_range = cfg["time_range"]
# var_def = cfg["var_def"]
# variables = cfg["variables"]
# xarray_kwargs = cfg["xarray_kwargs"]

# omsa.set_criteria(criteria)

# if readers is not None:
#     input_readers = readers
#     readers = []
#     if "local" in input_readers:
#         readers.append(odg.local)
#     if "erddap" in input_readers:
#         readers.append(odg.erddap)
#     if "axds" in input_readers:
#         readers.append(odg.axds)

# search = omsa.run(
#     approach=approach,
#     loc_model=loc_model,
#     axds=axds,
#     bbox=bbox,
#     criteria=criteria,
#     erddap=erddap,
#     figname_map=figname_map,
#     figname_data_prefix=figname_data_prefix,
#     local=local,
#     only_search=only_search,
#     only_searchplot=only_searchplot,
#     parallel=parallel,
#     readers=readers,
#     run_qc=run_qc,
#     skip_units=skip_units,
#     time_range=time_range,
#     stations=stations,
#     var_def=var_def,
#     variables=variables,
#     xarray_kwargs=xarray_kwargs,
# )
