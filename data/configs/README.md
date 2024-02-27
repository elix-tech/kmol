This file as for purpose to explain the recursive mechanism of the configuration files.

It can be a tedious task to repeat all the same configuration with minor changes for each experiment. 
As well as difficult to know what is different between various configuration.

We decided to introduce a recursive mechanism to the loading of the configuration file. 
This will enable the user to load an existing configuration and change only a few parameters. 
We have also added support for YAML configuration file. Let's see some example usage.

```yaml
__load__: ["../bc_01_config.json"]

epochs: 150
batch_size: 64
output_path: my_exp_dir
```

The recursive mechanism works by providing a new argument `__load__`. 
This argument expects a list of absolute path or relative path. The relative path 
is to start with the location of the configuration file. 
In the example `bc_01_config.json` will be in the parent folder of the used configuration file.

It is also possible to load multiple configuration files:

```yaml
__load__: [
  "model.yaml",
  "data.yaml"
]

epochs: 150
batch_size: 64
output_path: my_exp_dir
```

In this example we separated the model and the dataset configuration files. 
This way we can easily switch architecture or dataset. If there are some shared parameters 
in both `model.yaml` and `data.yaml` then the parameter of the later configuration 
file, here `data.yaml`, will be kept.

It can also be used to update only certain parts of a previous config, without changing the original file.

Let's look at this `model.yaml` config.

```yaml
# base_model.yaml
model:
    type: "protein_ligand"
    protein_module:
        type: "linear"
        in_features: 9723
        dropout: 0.1
        hidden_features: 256
        out_features: 128

    ligand_module:
        type: "graph_convolutional"
        in_features: 45
        out_features: 128
        hidden_features: 1024
        edge_features: 12
        dropout: 0.1
        layer_type: "torch_geometric.nn.LEConv"
        layers_count: 3
        molecule_features: 17
        is_residual: 1
        read_out: "attention"
        norm_layer: "kmol.model.layers.BatchNorm"
```

We can change the number of output features of the protein network without redefining a full configuration file.

```yaml
# new_experiment.yaml
__load__: ["model.yaml"]

 model:
     protein_module:
        out_features: 256
```

We don't need to define the full dictionary again. However, for list configuration the user will need to define the full list again.
The featurizers argument is an example of a list where we will need to provide the complete list in case of a change.

```yaml
# featurizer.yaml
# (Note that in yaml `-` indicate one element of a lists)
featurizers:
    - type: "graph"
      inputs: ["smiles"]
      outputs: ["ligand"]
      descriptor_calculator:
        type: "rdkit"

    - type: "bag_of_words"
      inputs: ["target_sequence"]
      outputs: ["protein"]
      should_cache: true
      vocabulary: ["A", "C", "D", "E", "F"]
      max_length: 3
```

```yaml
# This is not possible and will only update the overwrite the previous featurizer
__load__: ["featurizer.yaml"]
featurizers:
    - type: "graph"
      descriptor_calculator:
        type: "mordred"

```

To avoid unwanted overwriting of a model or a config, an extra folder is added in the output path directory. 
This directory will be named based on the time the program was launched. 
For example, if we enter the `output_path` `my_exp_dir` the result will be saved 
in a directory following the format `my_exp_dir/AAAA-MM-DD_HH-MM`. 
In doing so, we can keep track of the various runs done with the same configuration file or with the same output directory. 

The full configuration will also be saved in yaml and json format. So that the user can double check his recursive config worked correctly.
These files are generated at the start of any training. Note that the previous format without the `__load__` parameter will still work without any issue.
