![](docs/logo.png)
--------------------------------------------------------------------------------

kMoL is a machine learning library for drug discovery and life sciences, with federated learning capabilities.
Some of its features include state-of-the-art graph-based predictive models, explainable AI components, and differential privacy for data protection.
The library was benchmarked on datasets containing ADME properties (Absorption, Distribution, Metabolism, Excretion), toxicity, and binding affinities values.
 
Models are built using PyTorch and PyTorch Geometric.

## Installation
Cuda toolkit of at least 11.1 needs to be install.
Dependencies can be installed with conda:
```bash
conda env create -f environment.yml
conda activate kmol
bash install.sh
```

## Local Examples

All experiments are performed using configuration files (JSON).

A detailed documentation on how to write configuration files can be found under section 3.4 of `docs/documentation.pdf`.
Sample configurations can be found under `data/configs/model/`.

Each experiment starts with a dataset.
In these examples we focus on the [Tox21 Dataset](https://tripod.nih.gov/tox21/challenge/data.jsp) for which we define the experimental settings in `data/configs/model/tox21.json`.
After downloading the dataset to a suitable location, point to dataset with the "input_path" option in this JSON file.

### Training
The `train` command can be used to train a model. 
 
```bash
kmol train data/configs/model/tox21.json
```

### Finding the best checkpoint
Training will save a checkpoint for each individual epoch. 
These can be evaluated on a test split to find the best performing one with the `find_best_checkpoint` command.

```bash
kmol find_best_checkpoint data/configs/model/tox21.json
```

### Validate (a single checkpoint):
If a `checkpoint_path` is set in the JSON file for a specific checkpoint, it can be evaluated with the `eval` command. 

```bash
kmol eval data/configs/model/tox21.json
```

### Predict
Running inference is possible with the `predict` command.
This is performed on the test split by default.

```bash
kmol predict data/configs/model/tox21.json
```

A list of all available commands is available in the documentation.

## Federated Learning Examples

Similar to local training, a JSON configuration is needed to specify the training options.

In addition, a configuration file is needed for the server and each individual client to establish proper communication.
A detailed documentation on how to configure the server and clients can be found under section 3.5.1 and 3.5.2 of `docs/documentation.pdf` respectively.
Sample configurations can be found under `data/configs/mila/`.  

### Starting the server
The server should start before clients start connecting.

```bash
mila server data/configs/mila/naive_aggregator/tox21/clients/2/server.json
```

### Starting a client
Once the server is up, clients can join the federated learning process.
```bash
mila client data/configs/mila/naive_aggregator/tox21/clients/2/client1.json
```

Servers can be configured to wait for a specific number of clients.
Another client can be simulated from a new terminal:
```bash
mila client data/configs/mila/naive_aggregator/tox21/clients/2/client2.json
```

### Runing a Aggregation in Command Line.

Once all model have been run with the kmol module, it is possible to aggregate it using 
what we introduce as a script.

It is possible to use a new command line argument called `kmol-script`. This argument
expect only a config file containing all the necessary argument for that script.

In out case we are want to run an manual aggregation. So we can run:

```
kmol-script manual_aggregator.yaml
```

`manual_aggregator.yaml` is define as the following:

```yaml
script:
  type: "manual_aggregation"
  chekpoint_paths: 
    - data/logs/local/tester1/2022-10-20_17-10/checkpoint_10.pt
    - data/logs/local/tester2/2022-10-20_17-10/checkpoint_10.pt
    - data/logs/local/tester3/2022-10-20_17-10/checkpoint_10.pt
  aggregator_type: "mila.aggregators.WeightedTorchAggregator"
  aggregator_options:
    weights: [0.8, 0.1, 0.1]
  save_path: "data/logs/manual_aggregator/2.aggregator"

```

- `type`: Would be the type of script we want to run.
- `checkpoint_paths`: A list of checkpoint path we want to aggregate.
- `aggregator_type`: The type of aggregator to use.
- `aggregator_options`: The argument taken to instanciate the aggregator.

Note: that for WeightedTorchAggregator the weights argument is a bit different.
In mila we are xpecting a dictionary, here a list of weight is enough. The order of
the weights should follow the order of the checkpoint_paths

- `save_path`: Were to save the aggregator.

If other type of aggregator is needed, only `aggregator_type` and `aggregator_options` 
needs to be change.

You can find the aggregator and their argument in `src/mila/aggregators.py`