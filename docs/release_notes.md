# Release v2.1

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