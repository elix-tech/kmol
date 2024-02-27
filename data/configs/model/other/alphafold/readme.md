
In order to run the openfold model the follwing steps are needed:

# MSA Preparation

You can follow the openfold documentation https://github.com/aqlaboratory/openfold#usage

Or any otherway to download / generate the msa.

## Download the following in `data` foldet (or anywhere else if you change the configurations):

- Download alphafold db, which can be downloaded with this script (very large) `/home/vincent/kmol-internal/src/kmol/vendor/openfold/scripts/download_alphafold_dbs.sh`
- `openfold_params` can be downloaded with the following script (50GB) `src/kmol/vendor/openfold/scripts/download_openfold_params.sh`


## Launch the computation of MSA on the dataset.

Note: This step is particularly instable for reason outside our reach. We are
just providing and easy way to launching the script in an docker environment. 
In order to make sure the protein has been process, the file `bfd_uniclust_hits.a3m`
need to be present in the protein folder at the end of the computation. Otherwise
delete the file present and rerun the computation.


First build the docker image of the openfold operation with the following command.

```
make build-docker-openfold
```

Then you will need to download all the necessary `.fasta` file for your computation. You can then use the 
`msa_preprocessing.yaml` script in order to compute the msa.

```
kmol-script data/configs/model/other/alphafold/msa_preprocessing.yaml
```

# Launch the training

Since the msa are quiet heavy computation training and applying the inference of alphafold at
each step is generally to time consuming.
The `MsaFeaturizer` will compute the inference on alphafold. Since this steps takes 
a lot of memory the current step up launch it separatly to first save all the inference to a file 
before compute a network that uses the inference.

To launch the inference and save it to a file you can launch the following:

```
kmol featurize data/configs/model/other/alphafold/alphafold_preprocessing.json
```

Once this is done you can launch a training using those inference as featurization for the 
protein infromation with the following command.

```
kmol train data/configs/model/other/alphafold/alfafold_output_featurization_model.json
```


Note that we are loading the previously generated files with the following featurizer.

```
    {
    "type": "pickle_load",
    "inputs": [
        "protein_tag"
    ],
    "outputs": [
        "protein"
    ],
    "folder_path": "data/datasets/reduce_msa_featurizer_256/protein_tag",
    "suffix": ".pkl"
    }
```

