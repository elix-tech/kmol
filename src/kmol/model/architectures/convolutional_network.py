from typing import Dict, Any, List

import torch

from ..layers import LinearBlock
from .abstract_network import AbstractNetwork


class ConvolutionalNetwork(AbstractNetwork):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.convolutional_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(16),
            torch.nn.ReLU(),
        )
        self.last_hidden_layer_name = "linear_block.block.1"
        self.linear_block = LinearBlock(in_features=1880, hidden_features=hidden_features, out_features=out_features)

    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        x = data[self.get_requirements()[0]]

        x = self.convolutional_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_block(x)

        return x
