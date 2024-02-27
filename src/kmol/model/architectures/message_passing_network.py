from typing import Dict, Any, List

import torch
import torch_geometric as geometric

from kmol.model.architectures.abstract_network import AbstractNetwork


class MessagePassingNetwork(AbstractNetwork):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        edge_features: int,
        edge_hidden: int,
        steps: int,
        dropout: float = 0,
        aggregation: str = "add",
        set2set_layers: int = 3,
        set2set_steps: int = 6,
    ):
        super().__init__()
        self.out_features = out_features
        self.projection = torch.nn.Linear(in_features, hidden_features)

        edge_network = torch.nn.Sequential(
            torch.nn.Linear(edge_features, edge_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden, hidden_features * hidden_features),
        )

        self.convolution = geometric.nn.NNConv(hidden_features, hidden_features, edge_network, aggr=aggregation)
        self.gru = torch.nn.GRU(hidden_features, hidden_features)

        self.set2set = geometric.nn.Set2Set(hidden_features, processing_steps=set2set_steps, num_layers=set2set_layers)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features),
        )

        self.activation = torch.nn.ReLU()
        self.steps = steps
        self.last_hidden_layer_name = "mlp.1"

    def get_requirements(self) -> List[str]:
        return ["graph"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data[self.get_requirements()[0]]
        x = data.x.float()

        out = self.activation(self.projection(x))
        h = out.unsqueeze(0)

        for _ in range(self.steps):
            m = self.activation(self.convolution(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.mlp(out)

        return out
