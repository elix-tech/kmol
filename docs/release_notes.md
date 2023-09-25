# Release v0.1

- [FL-50] Support for more regression metrics. 6 new options supported for `train_metrics` and `test_metrics`:
    - `pearson` - Pearson correlation
    - `spearman` - Spearman correlation
    - `kl_div` - Kullback Leibler Divergence
    - `js_div` - Jensen–Shannon Divergence
    - `chebyshev` - Chebyshev distance
    - `manhattan` - Manhattan/CityBlock distance
- [FL-49] Molecules without any bonds between atoms will not be skipped anymore
- [FL-53] Stratified splitters can now split by input columns as well
- Multiple featurizers can target the same input/column multiple times now.
- 2 new commands added:
    - `kmol preload config.json` - will load/featurize/transform the dataset and cache it
    - `kmol splits config.json` - will print the indices included in each generated split
- [FL-61] Added a converter featurizer which can convert between all OpenBabel supported molecular formats (https://openbabel.org/wiki/Babel#File_Formats)
    - Example usage: `"featurizers": [{"type": "converter", "inputs": ["column_name_in_file", "can_be_more_than_one"], "outputs": ["desired_name_1", "desired_name_2"], "source_format": "inchi", "target_format": "smi"}]`
- 5 new splitters have been added:
    - `scaffold_balancer` - split the dataset by scaffold, trying to keep an equal number of samples for each scaffold/split
        - Example usage: `"splitter": {"type": "scaffold_balancer", "splits": {"train": 0.8, "test": 0.2}, "seed": 42}`
    - `scaffold_divider` - split the dataset by scaffold, trying to keep unique scaffolds in each split
        - Example usage: `"splitter": {"type": "scaffold_divider", "splits": {"train": 0.8, "test": 0.2}, "seed": 42}`
    - `butina_balancer` - a similarity based splitter which uses Butina clustering (tries to keep the splits similar)
        - Example usage: `"splitter": {"type": "butina_balancer", "splits": {"train": 0.8, "test": 0.2}, "butina_cutoff": 0.5, "fingerprint_size": 1024, "radius": 2}`
    - `butina_divider` -  a similarity based splitter which uses Butina clustering (tries to keep the splits dissimilar)
        - Example usage: `"splitter": {"type": "butina_divider", "splits": {"train": 0.8, "test": 0.2}, "butina_cutoff": 0.5, "fingerprint_size": 1024, "radius": 2}`
    - `descriptor` - performs a stratified split based on an RdKit descriptor. For a list of all supported descriptors, please visit the RdKit docs (https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors)
        - Example usage: `"splitter": {"type": "descriptor", "descriptor": "MolWt", "splits": {"train": 0.8, "test": 0.2}, "seed": 42, "bins_count": 10}`
- [FL-82] Added support for an RdKit style sketching mode in visualization modules (IntegratedGradients)
    - Example usage: `"visualizer": {"mapping_file_path": "visualization/tox21.csv", "targets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "sketcher": {"type": "rdkit", "output_path": "visualization/tox21/"}}`
- [FL-83] Added a mapping file (CSV) which links generated images to SMILES strings when visualization features are used.
    - This is configured with the `mapping_file_path` setting under the `visualizer` group.
- [FL-78] Added support for a one-hot transformer (so string based outputs can be converted to numeric values automatically).
    - Example usage: `"transformers": [{"type": "one_hot", "target": 0, "classes": ["Low", "High"]}]`
- [FL-81] Added a summary file which keeps track of Bayesian optimization trial results.
    - The summary report can be found in the specified `output_path` and is called `summary.csv`


Notes:
- This release has new dependencies. Please recreate your conda environment.
- Please delete your cache folder after updating.
- Older checkpoints might not be compatible with this release.


# Realease v0.2

- Added a `smiles_field` in splitters to allow dataset with smiles columns having a name different from `smiles`
    - Example usage: `"splitter": {"type": "scaffold_balancer", "splits": {"train": 0.8, "test": 0.2}, "seed": 42, "smiles_field": "column_name"}`

- FedProx Aggregation support
    - Fed prox paper: `https://arxiv.org/abs/1812.06127`
    - Example usage: `"observers": { "after_criterion": [{"type": "add_fedprox_regularization", "mu": 1}]}`
- Support for MC Dropout
- FASTAFeaturizer support

Notes:
- This release has new dependencies. Please recreate your conda environment.
- Please delete your cache folder after updating.
<!-- 




1.0.0



 -->


# Release 1.0.0 kMoL First tracked release

For retro compatibility use of the kmol library, we decide to start releasing tag version of the library in between each big update.
We will try to provide as informative patch notes as possible for each release.

Note that the binary file provided don't include all the dependencies, you will need to create an environment with the right packages (see README.md) to be able to run it.
In future release we might add support for it, but since there is a need for pytorch installation, which is dependent of the user software, for now we choose to exclude it.


<!-- 




1.1.4



 -->

#  1.1.4 Adding 2022 summer project and federated learning with box

Here are the release notes of kMoL version 1.1.4.
Multiple functionalities have been added to this release.
This release also includes fixes that resolve several issues from the initial pre-release.
## Highlights

### Library version updates

We have updated various packages.

Due to incompatibility with PyTorch and CUDA 12, we have now bundled the correct version of CUDA with the conda environment. The expected CUDA version should be CUDA >= 11.7.1

For other modules:
- `pytorch`: v`1.6.0` -> v`1.13.1`
- `pytorch_geometric`: v`1.6.3` -> v`2.2.0`

### Breaking change

Due to the update of `pytorch_geometric` to 2.X version, model trained with earlier 
A version of `pytorch_geometric` might have some issues. Especially the model which used
`torch_geometric.nn.LEConv` as a layer. We noticed that `torch_geometric.nn.LEConv` 
as a different inference starting `2.0` which lead to different results. 
We advise the user to retrain the model. If needed, we will implement a solution to convert older models to match the new version requirements.

### Configuration Update

It is now possible to reuse a template configuration file and modify only a few parameters.

One major change in this update regards the way the configuration file is being
processed. We found that it was a tedious task to repeat all the same configuration with minor changes for each experiment. It was difficult to know what had been changed from the previous configuration.
We decided to introduce a recursive mechanism to the loading of the configuration file. This is to load an existing configuration and change only a few parameters. We have also added support for YAML configuration file. 

Let’s see some example usage.



```yaml
__load__: ["../bc_01_config.json"]

epochs: 150
batch_size: 64
output_path: my_exp_dir
```

The recursive mechanism works by providing a new argument `__load__`. 
This argument expects a list of absolute path or relative path. 
The relative path expect to start at the location of the configuration file. 
In the example `bc_01_config.json` will be in the parent folder of the used configuration file.

It is possible to load multiple configuration files. In this case, if the parameters are conflicting, the priority will depend on the list order.

```yaml
__load__: [
 "model.yaml",
 "data.yaml"
]

epochs: 150
```

It is possible to update only certain parts of previous configuration files. 
Note that this feature will only work with dictionary inputs if changes where to be made in 
a list, the full list needs to be updated (in the `featurizers` field for example).

```yaml
__load__: ["model.yaml"]

 model:
 protein_module:
 out_features: 256

```

### Kmol scripts

For tasks unrelated to training and inference of a model, we added a new command 
`kmol-script` for now only two scripts are available:

- `generate_msa`: which is used with the openfold based model to generate msa inputs.
(see below)
- `integrated_gradient`: a explainability script to explain the weight of various parts of a model using the captum library.

The usage is similar to other configurations:

```yaml
script:
 type: "name_script"
 other_parameter:
```
### Preprocessing functionality

We implemented new ways to preprocess the data. So far all data was preprocess at the beginning of the training and stored in RAM. This enables fast training but has its limitation when training with very large datasets.
- Online preprocessing: Instead of storing the preprocessing data in RAM we will recompute the processing for each batch.
to use it set `online_preprocessing` to `True`. The number of workers for the preprocessing tasks will be based on `num_workers`

- Disk caching preprocessing: In cases where featurization is long and memory intensive, we also introduce 
a way to save the feature to disk during the training. Enables it with `preprocessing_use_disk` set to `True`
and `preprocessing_disk_dir` as the directory to write the cache files.

Both preprocessing can't be used at the same time.

### Graphormer

Most of the core functionalities of Graphormer have been implemented in kMol. Certain features
might be missing, for instance the input format has to be respected within kMol’s
framework to ensure consistency across experiments.

This includes:

- a new featurizer `graphormer`
- a new architecture `graphormer_encoder`, you can refer to [original graphormer documentation](https://graphormer.readthedocs.io/en/latest/Parameters.html#command-line-tools)
for more information about the parameters 
    - `pre_layernorm`: bool = False,
    - `num_atoms`: `int` = 512 * 9,
    - `num_in_degree`: `int` = 512,
    - `num_out_degree`: `int` = 512,
    - `num_edges`: `int` = 1024 * 3,
    - `num_spatial`: `int` = 512,
    - `num_edge_dis`: `int` = 128,
    - `edge_type`: `str` = "multi_hop",
    - `multi_hop_max_dist`: `int` = 5,
    - `num_classes`: `int` = 1,
    - `remove_head`: `bool` = False,
    - `dropout`: `float` = 0.1,
    - `attention_dropout`: `float` = 0.1,
    - `activation_dropout`: `float` = 0.0,
    - `encoder_ffn_embed_dim`: `int` = 4096,
    - `encoder_layers`: `int` = 6,
    - `encoder_attention_heads`: `int` = 8,
    - `encoder_embed_dim`: `int` = 1024,
    - `share_encoder_input_output_embed`: `bool` = False,
    - `apply_graphormer_init`: `bool` = False,
    - `activation`: `str` = `torch.nn.GELU`,
    - `encoder_normalize_before`: `bool` = True

*Disclaimer*: During our integration, we note that there were several limitations found in the original implementation of Graphormer. Notably, it supports a maximum of 512 nodes/atoms and can only support a maximum of 1 feature. Also, due to the requirement of memory usage, it's advisable to utilize  disk caching when performing featurizations.

### MSA based model

A model based on [Openfold](https://github.com/aqlaboratory/openfold) was added.
It makes use of MSA computation as input and uses the first part of the Alphafold 
network to extract features.

The usage is complex and we refer the user to the documentation provided to Kyodai.


### Uncertainty estimation

 Model's uncertainty estimation during inference techniques such as EDL, MC Dropout, LRODD, and Ensemble have been integrated. However, the configuration and setup for these techniques can be complex and involve multiple steps. We advise users to consult the documentation for guidance.


### Federated learning update

We added the possibility of using Box to communicate between client and server. 
This leads to the modification of the federated learning config. The read me has been updated to reflect such changes. 
A documentation for the usage of box has also been added in `docs/documentation_box.pdf`

### Various Addition

- Circular fingerprints: a new type of `descriptor_calculator` name `circular_fingerprint`

```yaml
featurizers:
- descriptor_calculator:
 type: circular_fingerprint
 inputs:
  - smiles
 outputs:
  - ligand
 type: graph
```

- Augmentation techniques: to use augmentations use the `augmentations` parameter
 - Augmentation base on [AugliChem](https://baratilab.github.io/AugLiChem/molecule.html)
 - Random atom mask
 - Random bond delete
```yaml
augmentations:
  - type: "random_atom_mask" % or "random_bond_delete"
 p: 0.3
 input_field: "ligand"
```

- Static augmentation: Additional data created through augmentation. This will be applied before the start of the training to 
increase the size of the datasets. So it is a stochastic operation.
    - Motif removals (based on [AugliChem](https://baratilab.github.io/AugLiChem/molecule.html))
    - Decoy selection: uses decoy as an additional training data based on the similarity with other ligand in the dataset.



- Explainability: possibility to use Integrated gradient technique to explain a Protein Ligand model contribution.

```
script:
type: "IntegratedGradient"
config_path: "model_captum.yaml"
```

### Minor changes

- The output directory has an additional abstraction based on the date. Before 
result will be save in `my_exp_dir`, now it will be in `my_exp_dir/AAA-MM-DD_HH-MM` to 
avoid overwriting previous results. We will also save a copy of the configuration files and 
additional logs for the training.

<!-- 




1.1.5



 -->

# kMol 1.1.5 Patch Notes

## Makefile support

Most of the install and build commands are now available through the make file:

```bash
=== Usage ===

create-env             create a kmol conda environment for the project
build-docker           build the docker container
build-docker-openfold  build openfold docker image, only required to generate the msa script
```

## Docker support 

Since the last update the docker was not working due to the addition of the need of
cuda support during the build of kMol. We have updated the docker and it is now working
with the current version.

## Fixes

- Added assertions for the multitask loader.
    - The `self._task_column_name` values must be list.
    - The `self._task_column_name` values must be between 0 and `max_num_tasks` -1.

- Fix the build of the environment. Due to local build with cuda support, the build
    could fail in some cases.
This should fix the following issue:
```bash
 [ ERROR | logger.py:78] > An unhandled exception occured: /home/s_ogata/pjt-kmol/kmol-1.1.3/src/attn_core_inplace_cuda.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda20CUDACachingAllocator9allocatorE.
```

- Fix issue in the `MsaFeaturizer`:
```bash
 [ ERROR  | logger.py:83] > An unhandled exception occured: 'index'
```


<!-- 




1.1.6



 -->


# kMol 1.1.6 Patch Notes

## Support for dynamic input in Convolutional Network

We added an argument in the Convolutional Network.
When used with the ProteinLigand Network we expected the following featurizer configuration
for the protein.

```json
{
    "type": "token",
    "inputs": ["target_sequence"],
    "outputs": ["protein"],
    "should_cache": true,
    "vocabulary": ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"],
    "max_length": 3011
},
{
    "type": "transpose",
    "inputs": ["protein"],
    "outputs": ["protein"]
}
```
Especially the `max_length` parameter was fixed to 3011. Now a new parameter `in_length`
was added to enable various `max_length`.

```json
"protein_module": {
    "type": "convolutional",
    "in_length": 3011,
    "in_features": 21,
    "hidden_features": 64,
    "out_features": 16
}
```

## Fixes

- Fixed an issue where existing conda environments conflict with a fresh installation of kMol.
- Various other fixes included in activation and deactivation of the conda environment
- EDL prediction score fix


# kMol 1.1.7 Patch Notes

## evidential output request

For the inference mode `evidential_classification_multilabel_nologits`, it now outputs the `softmax_score` and the `belief_mass` in the prediction file as columns.


## Fixes 

- Typo: g**rcp** was changed to g**rpc** to the places it was mistyped.