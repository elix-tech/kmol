#!/bin/bash

echo " - Setting up conda"
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate base

# Workaround cuda path prepended to conda on gcp instances
unset LD_LIBRARY_PATH
export PATH="$(dirname $(which python)):${PATH}"

ENV_NAME=$(grep "name:" environment.yml | cut -d: -f2)
LOCATION=${1:-""}

if (conda env list | cut -d ' ' -f1 | grep -q "^${ENV_NAME}$") && [ "$LOCATION" == "" ] ; then
    echo "The environment ${ENV_NAME} already created, stopping the generation"
    echo "If needed delete and recreate the environment to delete a conda env use: "
    echo "conda env remove -n ${ENV_NAME}"
    exit 1
fi

if [ "$LOCATION" != "" ] ; then
    ENV_NAME="$LOCATION"
    LOCATION="-p $LOCATION"
fi

echo " - Installing kMoL dependencies into $LOCATION"
conda env create -f environment.yml $LOCATION

# Set PYTHONNOUSERSITE to true for our environment, this allow the conda environnment
# effective insulation from Python packages installed at user-level.
conda activate $ENV_NAME
VENV_CONDA_PREFIX="$CONDA_PREFIX"
conda deactivate

echo " - Configuring kMoL virtual environment"
mkdir -p $VENV_CONDA_PREFIX/etc/conda/activate.d
(
    echo 'export KMOL_ORIG_LD="${LD_LIBRARY_PATH:-none}"';
    echo 'unset LD_LIBRARY_PATH';

    echo 'export KMOL_ADDED_PATH="$(dirname $(which python)):"';
    echo 'export PATH="${KMOL_ADDED_PATH}${PATH}"';

    echo 'export PYTHONNOUSERSITE=True'
) > $VENV_CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $VENV_CONDA_PREFIX/etc/conda/deactivate.d
(
    echo '[ "${KMOL_ORIG_LD}" != "none" ] && export LD_LIBRARY_PATH="${KMOL_ORIG_LD}"';
    echo 'unset KMOL_ORIG_LD';

    echo 'export PATH="$(echo "${PATH}"|sed "s,${KMOL_ADDED_PATH},,")"';
    echo 'unset KMOL_ADDED_PATH';

    echo 'unset PYTHONNOUSERSITE'
) > $VENV_CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda activate $ENV_NAME
rm -f src/*.so

echo " - Installing kMoL"
# Install local package
pip install --no-build-isolation -e .

echo
echo
echo " ==> Installation done: run 'conda activate kmol' to enable the virtual env and be able to run the kmol command."
