import torch
from typing import Dict, Any, List

from kmol.model.layers import LinearBlock
from kmol.model.architectures.abstract_network import AbstractNetwork


class LinearNetwork(AbstractNetwork, LinearBlock):
    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        features = data[self.get_requirements()[0]]
        return super().forward(features)
