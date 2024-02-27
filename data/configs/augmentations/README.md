

Augmentation are optional configuration. We suggest the user to define an augmentation file to add it later on with the `__load__` parameter (see below).

There are 2 type of augmentations `augmentation` and `static_augmentation` which define motif removal and decoy selection. 
The separation is based on the implementation. `augmentation` are applied during training and have a stochastic aspect to them. 
Whereas `static_augmentation` are generated during the preprocessing of the dataset, and will be added to the main dataset, 
they will stay the same accros the training. It can be seen as additional data.


The following config if ran will launch the model with the random atom mask augmentation.

```yaml
__load__: [
  "data/configs/model/plb/fingerprint_ligand+bow_protein.json",
  "/home/vincent/kmol-internal/data/configs/augmentations/random_atom_mask.yaml"
]
```