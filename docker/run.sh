#!/bin/bash --login

source activate federated
python -c "import torch;print(torch.__version__)"


