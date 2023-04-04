#!/bin/env bash

eval "$(/opt/elix/anaconda3/setup)"
enable-conda
conda activate kmol
pip install --no-build-isolation .