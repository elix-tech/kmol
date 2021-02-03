from typing import Optional

import torch

from lib.core.helpers import SuperFactory


class GraphConvolutionWrapper(torch.nn.Module):

    def __init__(
            self, in_features: int, out_features: int, dropout: float, layer_type: str = "torch_geometric.nn.GCNConv",
            is_residual: bool = True, norm_layer: Optional[str] = None, activation: str = "torch.nn.ReLU", **kwargs
    ):
        super().__init__()

        self.convolution = SuperFactory.reflect(layer_type)(in_features, out_features, **kwargs)
        self.norm_layer = SuperFactory.reflect(norm_layer)(out_features) if norm_layer else None
        self.residual_layer = torch.nn.Linear(in_features, out_features) if is_residual else None
        self.activation = SuperFactory.reflect(activation)()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.convolution(x, edge_index)

        if self.residual_layer:
            x += self.residual_layer(identity)

        if self.norm_layer:
            x = self.norm_layer(x)

        x = self.activation(x)
        x = self.dropout(x)

        return x
