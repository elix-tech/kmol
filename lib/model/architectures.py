from typing import Dict, Any, Optional

import torch
import torch_geometric as geometric

from lib.model.layers import GraphConvolutionWrapper, TripletMessagePassingLayer
from math import floor


class AbstractNetwork(torch.nn.Module):
    pass


class GraphConvolutionalNetwork(AbstractNetwork):

    def __init__(
            self, in_features: int, hidden_features: int, out_features: int, molecule_features: int,
            dropout: float, layer_type: str = "torch_geometric.nn.GCNConv", layers_count: int = 2,
            is_residual: bool = True, norm_layer: Optional[str] = None, activation: str = "torch.nn.ReLU", **kwargs
    ):
        super().__init__()

        self.convolutions = torch.nn.ModuleList()
        self.convolutions.append(GraphConvolutionWrapper(
            in_features=in_features, out_features=hidden_features, dropout=dropout, layer_type=layer_type,
            is_residual=is_residual, norm_layer=norm_layer, activation=activation, **kwargs
        ))

        for _ in range(layers_count - 1):
            self.convolutions.append(GraphConvolutionWrapper(
                in_features=hidden_features, out_features=hidden_features, dropout=dropout, layer_type=layer_type,
                is_residual=is_residual, norm_layer=norm_layer, activation=activation, **kwargs
            ))

        self.molecular_head = torch.nn.Sequential(
            torch.nn.Linear(molecule_features, hidden_features // 4),
            torch.nn.Dropout(p=min(hidden_features / in_features, 0.7)),
            torch.nn.BatchNorm1d(hidden_features // 4),
            torch.nn.ReLU()
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(floor(2.25 * hidden_features), hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features)
        )

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data["graph"]
        x = data.x.float()

        for convolution in self.convolutions:
            x = convolution(x, data.edge_index, data.edge_attr, data.batch)

        max_pool_output = geometric.nn.global_max_pool(x, batch=data.batch)
        add_pool_output = geometric.nn.global_add_pool(x, batch=data.batch)
        molecule_features = self.molecular_head(data.molecule_features)

        x = torch.cat((max_pool_output, add_pool_output, molecule_features), dim=1)
        x = self.mlp(x)

        return x


class MessagePassingNetwork(AbstractNetwork):

    def __init__(
            self, in_features: int, hidden_features: int, out_features: int,
            edge_features: int, edge_hidden: int, steps: int, dropout: float = 0,
            aggregation: str = "add", set2set_layers: int = 3, set2set_steps: int = 6
    ):
        super().__init__()

        self.projection = torch.nn.Linear(in_features, hidden_features)

        edge_network = torch.nn.Sequential(
            torch.nn.Linear(edge_features, edge_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden, hidden_features * hidden_features)
        )

        self.convolution = geometric.nn.NNConv(hidden_features, hidden_features, edge_network, aggr=aggregation)
        self.gru = torch.nn.GRU(hidden_features, hidden_features)

        self.set2set = geometric.nn.Set2Set(hidden_features, processing_steps=set2set_steps, num_layers=set2set_layers)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features)
        )

        self.activation = torch.nn.ReLU()
        self.steps = steps

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data["graph"]
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


class TripletMessagePassingNetwork(AbstractNetwork):

    def __init__(
            self, in_features: int, hidden_features: int, out_features: int, edge_features: int,
            layers_count: int, dropout: float = 0, set2set_layers: int = 1, set2set_steps: int = 6
    ):
        super().__init__()
        self.dropout = dropout
        self.projection = torch.nn.Linear(in_features, hidden_features)

        self.message_passing_layers = torch.nn.ModuleList([
            TripletMessagePassingLayer(hidden_features, edge_features) for _ in range(layers_count)
        ])

        self.set2set = geometric.nn.Set2Set(hidden_features, processing_steps=set2set_steps, num_layers=set2set_layers)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features)
        )

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data["graph"]
        x = data.x.float()

        out = self.projection(x)
        out = torch.nn.functional.celu(out)

        edge_attr = data.edge_attr.float()
        for message_passing_layer in self.message_passing_layers:
            out = out + torch.nn.functional.dropout(
                message_passing_layer(out, data.edge_index, edge_attr),
                p=self.dropout,
                training=self.training
            )

        out = torch.nn.functional.dropout(
            self.set2set(out, data.batch),
            p=self.dropout,
            training=self.training
        )

        out = self.mlp(out)
        return out


class ConvolutionalProteinLigandNetwork(AbstractNetwork):

    def __init__(self, ligand_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()

        self.protein_convolution = torch.nn.Conv1d(in_channels=21, out_channels=10, kernel_size=3, stride=1)
        self.protein_max_pooling = torch.nn.MaxPool1d(16)

        self.protein_module = torch.nn.Sequential(
            torch.nn.Linear(1880, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True),
            torch.nn.ReLU(),
        )

        self.ligand_module = torch.nn.Sequential(
            torch.nn.Linear(ligand_features, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True),
            torch.nn.ReLU(),
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(32, 16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_features, bias=True),
        )

        self.protein_module.apply(self._init_weights)
        self.ligand_module.apply(self._init_weights)
        self.output_module.apply(self._init_weights)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def _init_weights(self, layer: torch.nn) -> None:
        if type(layer) == torch.nn.Linear:
            layer.weight.data.copy_(
                torch.nn.init.xavier_uniform_(layer.weight.data)
            )

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        ligand_features = data["ligand"].float()
        protein_features = data["protein"].float()

        protein_convolution = self.protein_convolution(protein_features)
        protein_convolution = self.activation(protein_convolution)

        protein_max_pooling = self.protein_max_pooling(protein_convolution)
        protein_max_pooling = self.activation(protein_max_pooling)

        protein_flatten = torch.flatten(protein_max_pooling, start_dim=1)
        protein_features = self.protein_module(protein_flatten)
        ligand_features = self.ligand_module(ligand_features)

        combined_features = torch.cat((protein_features, ligand_features), dim=-1)
        output = self.output_module(combined_features)

        return output
