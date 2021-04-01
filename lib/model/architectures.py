from abc import ABCMeta, abstractmethod
from math import floor
from typing import Dict, Any, Optional, List

import torch
import torch_geometric as geometric

from lib.model.layers import GraphConvolutionWrapper, TripletMessagePassingLayer, LinearBlock


class AbstractNetwork(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def get_requirements(self) -> List[str]:
        raise NotImplementedError

    def map(self, module: "AbstractNetwork", *args) -> Dict[str, Any]:
        requirements = module.get_requirements()

        if len(args) != len(requirements):
            raise AttributeError("Cannot map inputs to module")

        return {requirements[index]: args[index] for index in range(len(requirements))}


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

        self.molecular_head = lambda x: torch.Tensor()
        if molecule_features:
            self.molecular_head = torch.nn.Sequential(
                torch.nn.Linear(molecule_features, hidden_features // 4),
                torch.nn.Dropout(p=min(hidden_features / in_features, 0.7)),
                torch.nn.BatchNorm1d(hidden_features // 4),
                torch.nn.ReLU()
            )

        mlp_features = 2 + 0.25 * bool(molecule_features)
        mlp_features = floor(mlp_features * hidden_features)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features)
        )

    def get_requirements(self) -> List[str]:
        return ["graph"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data[self.get_requirements()[0]]
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
                training=self.training
            )

        out = torch.nn.functional.dropout(
            self.set2set(out, data.batch),
            p=self.dropout,
            training=self.training
        )

        out = self.mlp(out)
        return out


class LinearNetwork(AbstractNetwork, LinearBlock):

    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        features = data[self.get_requirements()[0]]
        return super().forward(features)


class ConvolutionalNetwork(AbstractNetwork):

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()

        self.convolutional_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(16),
            torch.nn.ReLU()
        )

        self.linear_block = LinearBlock(in_features=1880, hidden_features=hidden_features, out_features=out_features)

    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        x = data[self.get_requirements()[0]]

        x = self.convolutional_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_block(x)

        return x


class ProteinLigandNetwork(AbstractNetwork):

    def __init__(
            self, protein_module: AbstractNetwork, ligand_module: AbstractNetwork,
            hidden_features: int, out_features: int
    ):
        super().__init__()

        self.protein_module = protein_module
        self.ligand_module = ligand_module
        self.output_module = LinearBlock(hidden_features, 16, out_features)

        self.protein_module.apply(self._init_weights)
        self.ligand_module.apply(self._init_weights)
        self.output_module.apply(self._init_weights)

        self.activation = torch.nn.ReLU()

    def _init_weights(self, layer: torch.nn) -> None:
        if type(layer) == torch.nn.Linear:
            layer.weight.data.copy_(
                torch.nn.init.xavier_uniform_(layer.weight.data)
            )

    def get_requirements(self) -> List[str]:
        return ["ligand", "protein"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        requirements = self.get_requirements()

        ligand_features = self.map(self.ligand_module, data[requirements[0]])
        protein_features = self.map(self.protein_module, data[requirements[1]])

        protein_features = self.activation(self.protein_module(protein_features))
        ligand_features = self.activation(self.ligand_module(ligand_features))

        combined_features = torch.cat((protein_features, ligand_features), dim=-1)
        output = self.output_module(combined_features)

        return output
