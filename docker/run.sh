#!/bin/bash --login

source activate kmol
python -c "import torch;print(torch.__version__)"
