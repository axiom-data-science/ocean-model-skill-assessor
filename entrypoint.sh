#!/usr/bin/bash
# Determine appropriate action for running the container:
# A) Run as in command line mode (-c)
# B) Run running Jupyter lab (-j)
#
# Both modes require the container to be started with the appropriate mounts to save the data.

set -o errexit
set -o nounset
set -o pipefail

function handle_exit() {
    echo "Something went wrong. Exiting..."
}
trap handle_exit SIGHUP SIGINT SIGQUIT SIGABRT SIGTERM

function usage {
    echo "Usage: $(basename $0) [-jc]" 2>&1
    echo '  -j, run JupyterLab'
    echo '  -c </path/to/config-file>, run in command line mode'
    exit 1
}
if [[ ${#} -eq 0 ]]; then
    usage
fi

optstring=":jc"
while getopts ${optstring} arg; do
    case $arg in
        j) JUPYTER_LAB=true;;
        c) JUPYTER_LAB=false;;
        \?) usage;;
    esac
done

shift

if [ "$JUPYTER_LAB" == true ]; then
    echo "Running JupyterLab.  Ensure port 8888 is open."
    /env/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/demos
else
    echo "Running in command line mode."
    /env/bin/python /omsa/ocean_model_skill_assessor/CLI.py "$@"
fi

echo "Ensure container is run with required options:"
echo "  1. mounts for data are available in the container (e.g. docker run -v /abs/path/to/data:/data)."