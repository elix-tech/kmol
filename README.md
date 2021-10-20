# Federated Learning Library (v2.2 nightly)

This file is outdated. Please use the documentation PDF for reference.

## Installation

Dependencies can be installed with conda:
```bash
conda env create -f environment.yml
conda activate kmol
bash install.sh
```

## Commands

Training:
```bash
kmol train data/configs/gcn.json
```

Validate (a single checkpoint):
```bash
kmol eval data/configs/gcn.json
```

Validate (all checkpoint):
```bash
kmol analyze data/configs/gcn.json
```

Inference:
```bash
kmol predict data/configs/gcn.json
```

