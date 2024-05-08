![](docs/logo.png)
--------------------------------------------------------------------------------

kMoL is a machine learning library for drug discovery and life sciences, with federated learning capabilities.
Some of its features include state-of-the-art graph-based predictive models, explainable AI components, and differential privacy for data protection.
The library was benchmarked on datasets containing ADME properties (Absorption, Distribution, Metabolism, Excretion), toxicity, and binding affinities values.

Models are built using PyTorch and PyTorch Geometric.

## Installation

kMoL can be used by installing it in a conda virtual environment or in a docker container. The following sections will describe how to install/use kMoL in each
installation mode. You can choose whichever better fit your needs.

### Using a conda virtual environment

This kind of installation is usually better if you already have conda installed or if you want to tweak kMoL code.

```bash
make create-env
conda activate kmol
```

You can then run `kmol` commands in the current shell and code/configurations changes will be picked up immediately.
Refer to the 'kMoL usage' section below for parameters

### Using a docker container

A docker image will be created with a miniforge installation and the current version of kMoL. Beside docker and working gpu drivers
on the host, it will be self contained and can be used by multiple users. On the other hand, if you want to tweak the code, you'll need to rebuild the
images after each modification: for this it is more efficient to install kMoL in a virtual environment on the host (eventually switching to the
docker installation to share your version)

```bash
make build-docker
```

Once an image with the current version is created, you can run kmol by passing the job/config to the 'docker run' command like this:

```bash
# Simplest command, will run 'kmol {job} {path_to_config}' in a container
docker run --rm -ti --gpus=all -e KMOL_UID=$(id -u) -e KMOL_GID=$(id -g) -v ./data:/home/kmol/data elix-kmol:1.1.9.2 {job} {path_to_config}

# Running without a parameter will start an interactive shell in the same environment then manually launching commands is also possible
docker run --rm -ti --gpus=all -e KMOL_UID=$(id -u) -e KMOL_GID=$(id -g) -v ./data:/home/kmol/data elix-kmol:1.1.9.2
```

Note: The configuration files provided are intended to be used with the 'conda virtual environment' installation where kMoL is installed
in development mode. When using the docker image, you'll have to adjust the paths in the config files accordingly. For example, in the
configuration file "tox21.json", the "loader">"input_path" value is set to "data/datasets/tox21.csv": in the docker container this would point
to an incorrect location. Replace it with "/home/kmol/data/datasets/tox21.csv", matching the volume bind of the docker run command.

## kMoL usage

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

### Box and Grpc parameters

There are two ways to run a federated example. One is with the grpc protocol to connect the client directly to the server. The second way uses box applications and sends the models to a box shared directory.  There is a needed set up to be done on Box for it to work. The set up won't be explained here, see `docs/box_documentation.pdf` for more details.

The grpc parameter should be contain in `grpc_configuration` like the following:

```json
  "server_type": "mila.services.servers.GrpcServer",
  "server_manager_type": "mila.services.server_manager.GrpcServicer",
  ...
  "grpc_configuration": {
    "target": "localhost:8024",

    "options": [
          ["grpc.max_send_message_length", 1000000000],
          ["grpc.max_receive_message_length", 1000000000],
          ["grpc.ssl_target_name_override", "localhost"]
      ],

    "use_secure_connection": false,
    "ssl_private_key": "data/certificates/client.key",
    "ssl_cert": "data/certificates/client.crt",
    "ssl_root_cert": "data/certificates/rootCA.pem"
  }
```

The client configuration only changes having the parameter `client_type` to `mila.services.clients.GrpcClient` istead of `server_type` and `server_manager_type`.

Note: if the user want to leave the default parameter it should still provide an empty directory to the grpc_configuration configuration.


As for box we will have a similar config type:


```json
  "server_type": "mila.services.servers.BoxServer",
  "server_manager_type": "mila.services.server_manager.BoxServicer",
  ...
  "box_configuration": {
    "box_configuration_path": "example_jwt_config.json",
    "shared_dir_name": "example-folder-jwt",
    "save_path": "my_path_inside_shared_dir_name",
    "group_name": "jwt-application-group-name"
  }
```

Similar to grpc the client config will only need `client_type` set to `mila.services.clients.BoxClient`

- `box_configuration_path`: The Public / private key pair file downloaded during the admin set up,
- `shared_dir_name`: The name of the directory shared with the group of the application,
- `save_path`: The path inside shared_dir_name where the training logs and weights should be save. To avoid issue we also add a date_time layer based on the time launched.,
- `group_name`: The name of the group of the application


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
expects only a config file containing all the necessary argument for that script.

In out case we are want to run a manual aggregation. So we can run:

```
kmol-script data/configs/manual_aggregator.yaml
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
