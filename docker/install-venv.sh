#!/bin/env -S bash --login

cd /opt/src

echo " -o- Activate kmol environment"
conda activate kmol

echo " -o- Installing kmol package"
pip install --no-build-isolation .

cd /opt
rm -rf src
