from typing import Dict, Any, List
import math

import torch

from kmol.model.layers import LinearBlock
from kmol.model.architectures.abstract_network import AbstractNetwork


class ConvolutionalNetwork(AbstractNetwork):
    def __init__(self, in_features: int, in_length: int, hidden_features: int, out_features: int):
        """
        This Network is expected to be use with a Protein-Ligand Network as a protein_module.
        The most common case is to use this with two featurizer a tokenizer and a transpose.
        Given an input vector of shape [bs, in_features, in_length]
        in_length: int: the length of the signal, for the most common case it will be max_length
        in_features: int: the channel dimension, for the most common case it will be the length of the vocabulary
        hidden_features: int: hidden dimension of the linear block
        out_features: dimension of the output tensor [bs, out_features]
        """
        super().__init__()
        self.out_features = out_features
        self.convolutional_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(16),
            torch.nn.ReLU(),
        )
        self.last_hidden_layer_name = "linear_block.block.1"
        in_features = math.floor((in_length - 2) / 16) * 10
        self.linear_block = LinearBlock(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        x = data[self.get_requirements()[0]]

        x = self.convolutional_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_block(x)

        return x
