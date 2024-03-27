from typing import List
import torch

from kmol.model.architectures.abstract_network import AbstractNetwork
from kmol.core.helpers import SuperFactory


class SequentialNetwork(AbstractNetwork):
    """Wrapper around pytorch Sequential"""

    def __init__(self, layers: List[AbstractNetwork]):
        super().__init__()
        self.layers = [SuperFactory.create(AbstractNetwork, layer) for layer in layers]
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def get_requirements(self):
        pass
