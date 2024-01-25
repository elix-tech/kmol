from typing import Dict, Any, List

import torch
import torch_geometric as geometric

from kmol.model.layers import TripletMessagePassingLayer
from kmol.model.architectures.abstract_network import AbstractNetwork


class TripletMessagePassingNetwork(AbstractNetwork):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        edge_features: int,
        layers_count: int,
        dropout: float = 0,
        set2set_layers: int = 1,
        set2set_steps: int = 6,
    ):
        super().__init__()
        self.out_features = out_features
        self.dropout = dropout
        self.projection = torch.nn.Linear(in_features, hidden_features)

        self.message_passing_layers = torch.nn.ModuleList(
            [TripletMessagePassingLayer(hidden_features, edge_features) for _ in range(layers_count)]
        )

        self.set2set = geometric.nn.Set2Set(hidden_features, processing_steps=set2set_steps, num_layers=set2set_layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features),
        )
        self.last_hidden_layer_name = "mlp.2"

    def get_requirements(self) -> List[str]:
        return ["graph"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data[self.get_requirements()[0]]
        x = data.x.float()

        out = self.projection(x)
        out = torch.nn.functional.celu(out)

        edge_attr = data.edge_attr.float()
        for message_passing_layer in self.message_passing_layers:
            out = out + torch.nn.functional.dropout(
                message_passing_layer(out, data.edge_index, edge_attr),
                p=self.dropout,
                training=self.training,
            )

        out = torch.nn.functional.dropout(self.set2set(out, data.batch), p=self.dropout, training=self.training)

        out = self.mlp(out)
        return out
