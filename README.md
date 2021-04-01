# Federated Learning Library

This file is outdated. Please use the documentation PDF for reference.

## Installation

Use the provided conda snapshot:
```bash
conda env create -f environment.yml  # first time only
conda activate federated
bash install_additional_dependencies.sh  # first time only
```

## Commands

Training:
```bash
python run.py train data/configs/gcn.json
```

Validate (a single checkpoint):
```bash
python run.py eval data/configs/gcn.json
```

Validate (all checkpoint):
```bash
python run.py analyze data/configs/gcn.json
```

Inference:
```bash
python run.py predict data/configs/gcn.json
```

