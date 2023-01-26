"""
Command Line Interface.
"""

import argparse

import cf_pandas as cfp

import ocean_model_skill_assessor as omsa


def is_int(s):
    """Check if string is actually int."""
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def is_float(s):
    """Check if string is actually float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    """With can user can input dicts on CLI."""

    def __call__(self, parser, namespace, values, option_string=None):
        """With can user can input dicts on CLI."""
        setattr(namespace, self.dest, dict())
        for value in values:
            # maxsplit helps in case righthand side of input has = in it, like filenames can have
            key, value = value.split("=", maxsplit=1)
            # catch list case
            if value.startswith("[") and value.endswith("]"):
                # if "[" in value and "]" in value:
                value = value.strip("][").split(",")
            # change numbers to numbers but with attention to decimals and negative numbers
            if is_int(value):
                value = int(value)
            elif is_float(value):
                value = float(value)
            getattr(namespace, self.dest)[key] = value


def main():
    """Parser method."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        help="What action to take? Options: make_catalog, proj_path, vocabs, run.",
    )
    parser.add_argument(
        "--project_name",
        help="All saved items will be stored in a subdirectory call `project_name` in the user application cache.",
    )

    # make_catalog options
    parser.add_argument(
        "--catalog_type",
        help="Which type of catalog to make? Options: erddap, axds, local.",
    )

    parser.add_argument(
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Input keyword arguments for the catalog. Available options are specific to the `catalog_type`. Dictionary-style input. More information on options can be found in `omsa.main.make_catalog` docstrings. Format for list items is e.g. standard_names='[sea_water_practical_salinity,sea_water_temperature]'.",
    )

    parser.add_argument(
        "--kwargs_search",
        nargs="*",
        action=ParseKwargs,
        help="Input keyword arguments for the search specification. Dictionary-style input. More information on options can be found in `omsa.main.make_catalog` docstrings. Format for list items is e.g. standard_names='[sea_water_practical_salinity,sea_water_temperature]'.",
    )

    parser.add_argument(
        "--vocab_name", help="Vocab file name, must be in the vocab user directory."
    )

    parser.add_argument(
        "--catalog_name", help="Catalog name, with or without suffix of yaml."
    )
    parser.add_argument("--description", help="Catalog description.")

    parser.add_argument(
        "--vocab_names",
        nargs="*",
        help="Name of vocabulary file, must be in the vocab user directory.",
    )

    parser.add_argument(
        "--verbose",
        help="Options are --verbose or --no-verbose. Print useful runtime commands to stdout if True as well as save in log, otherwise silently save in log.",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--mode", help="File mode for log file.", default="w")

    # run options
    parser.add_argument(
        "--catalog_names",
        nargs="*",
        help="Which catalogs, by name, to use? For example: catalog1 catalog2",
    )
    parser.add_argument(
        "--key", help="Key from vocab representing the variable to compare."
    )
    parser.add_argument(
        "--model_name",
        help="Name of catalog for model output, created in a `make_Catalog` command.",
    )
    parser.add_argument(
        "--ndatasets",
        type=int,
        help="Max number of datasets from input catalog(s) to use.",
    )

    parser.add_argument(
        "--kwargs_open",
        nargs="*",
        action=ParseKwargs,
        help="Input keyword arguments to be passed onto xarray open_mfdataset or pandas read_csv.",
    )

    parser.add_argument(
        "--metadata",
        nargs="*",
        action=ParseKwargs,
        help="Metadata to be passed into catalog.",
    )

    parser.add_argument(
        "--kwargs_map",
        nargs="*",
        action=ParseKwargs,
        help="Input keyword arguments to be passed onto map plot.",
    )

    args = parser.parse_args()

    # Make a catalog.
    if args.action == "make_catalog":
        omsa.make_catalog(
            catalog_type=args.catalog_type,
            project_name=args.project_name,
            catalog_name=args.catalog_name,
            description=args.description,
            metadata=args.metadata,
            kwargs=args.kwargs,
            kwargs_search=args.kwargs_search,
            kwargs_open=args.kwargs_open,
            vocab=args.vocab_name,
            save_cat=True,
            verbose=args.verbose,
            mode=args.mode,
        )

    # Print path for project name.
    elif args.action == "proj_path":
        print(omsa.PROJ_DIR(args.project_name))

    # Print available vocabularies.
    elif args.action == "vocabs":
        print([path.stem for path in omsa.VOCAB_DIR.glob("*")])

    # Print variable keys in a vocab.
    elif args.action == "vocab_info":
        vpath = omsa.VOCAB_PATH(args.vocab_name)
        vocab = cfp.Vocab(vpath)
        print(f"Vocab path: {vpath}.")
        print(f"Variable nicknames in vocab: {list(vocab.vocab.keys())}.")

    # Run model-data comparison.
    elif args.action == "run":
        omsa.main.run(
            project_name=args.project_name,
            catalogs=args.catalog_names,
            vocabs=args.vocab_names,
            key_variable=args.key,
            model_name=args.model_name,
            ndatasets=args.ndatasets,
            kwargs_map=args.kwargs_map,
            verbose=args.verbose,
            mode=args.mode,
        )
