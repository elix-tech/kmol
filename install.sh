#!/bin/env bash
eval "$(conda 'shell.bash' 'hook' 2> /dev/null)"
conda deactivate
ENV_NAME=$(grep "name:" environment.yml | cut -d: -f2)
# ENV_NAME="./env-test"
(conda env list | grep -q ${ENV_NAME}) && (echo "env ${ENV_NAME} already created.."; exit 1)
conda env create -f environment.yml

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