#!/bin/bash
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
ENV_NAME=$(grep "name:" environment.yml | cut -d: -f2)
LOCATION=${1:-""}

if (conda env list | grep -q ${ENV_NAME}) && [ "$LOCATION" == "" ] ; then
    echo "The environment ${ENV_NAME} already created, stopping the generation"
    echo "If needed delete and recreate the environment to delete a conda env use: "
    echo "conda env remove -n ${ENV_NAME}"
    exit 1
fi

if [ "$LOCATION" != "" ] ; then
    ENV_NAME="$LOCATION"
    LOCATION="-p $LOCATION"
fi

echo conda env create -f environment.yml $LOCATION
conda env create -f environment.yml $LOCATION

# Set PYTHONNOUSERSITE to true for our environment, this allow the conda environnment
# effective insulation from Python packages installed at user-level.
conda activate $ENV_NAME

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=True' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset PYTHONNOUSERSITE' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Deactivate / activate to set PYTHONNOUSERSITE=True
conda deactivate

conda activate $ENV_NAME

# Install local package
pip install --no-build-isolation -e .
