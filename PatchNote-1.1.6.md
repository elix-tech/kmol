

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
    "in_lenght": 3011,
    "in_features": 21,
    "hidden_features": 64,
    "out_features": 16
}
```