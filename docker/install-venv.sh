#!/bin/env -S bash --login

echo " -o- Activate kmol environment"
conda activate kmol

echo " -o- Installing kmol package"
pip install --no-build-isolation .

chmod 755 /opt/envs/kmol/pkgs/cuda-toolkit
