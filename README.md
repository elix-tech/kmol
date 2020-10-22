# Soon to be Federated Learning Library

For now, it only contains some graph based models.

## Installation

Use the provided conda snapshot:
```bash
conda env create -f environment.yml  # first time only
conda activate federated
```

## Data

- Tox21 (automatically downloaded)
- ChEMBL Protein-ligand activity (data/input/chembl/)

## Configuration

You need a JSON configuration file for each experiment you run. The options are:

- **model**: What model to use? [Available Options: "GraphConvolutionalNetwork", "GraphIsomorphismNetwork"]
- **data_loader**: Which data loader to use? [Available Options: "MoleculeNetLoader"]
- **dataset**: Which dataset should "MoleculeNetLoader" download? The dataset will be saved to "input_path". [Available Options: "tox21", "pcba", "muv", "hiv", "bbbp", "toxcast", "sider", "clintox"]

- **input_path**: Where is the input data?
- **output_path**: Where to save checkpoints?
- **checkpoint_path**: Which checkpoint to use during evaluation/inference?

- **epochs**: How many epochs to train?
- **batch_size**: ...
- **learning_rate**: ...
- **weight_decay**: ...
- **dropout**: ...
- **hidden_layer_size**: Hidden layer size for GCN and GIN.

- **threshold**: Inference threshold for classification.
- **train_ratio**: Between 0 and 1; used when creating data splits.
- **split_method**: How to split the data? [Available options: "index", "random"]
- **seed**: For data splits

- **use_cuda**: Should use GPU acceleration? (if not available, CPU will be used)
- **enabled_gpus**: Which GPUS to use?

- **log_level**: How many logs would you like to see? [Options: "debug", "info", "warn", "error", "critical"]
- **log_format**: Add additional information like timestamps to your logs.
- **log_frequency**: After how many iterations to print an update (while training)


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