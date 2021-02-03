from typing import Dict, Any, Optional

import torch
import torch_geometric as geometric

from lib.model.layers import GraphConvolutionWrapper
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
            x = convolution(x, data.edge_index)

        max_pool_output = geometric.nn.global_max_pool(x, batch=data.batch)
        add_pool_output = geometric.nn.global_add_pool(x, batch=data.batch)
        molecule_features = self.molecular_head(data.molecule_features)

        x = torch.cat((max_pool_output, add_pool_output, molecule_features), dim=1)
        x = self.mlp(x)

        return x


class GraphIsomorphismNetwork(AbstractNetwork):

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float):

        super().__init__()

        self.convolution_1 = self.__create_gin_convolution(in_features, hidden_features)
        self.convolution_2 = self.__create_gin_convolution(hidden_features, hidden_features)
        self.convolution_3 = self.__create_gin_convolution(hidden_features, hidden_features)
        self.convolution_4 = self.__create_gin_convolution(hidden_features, hidden_features)
        self.convolution_5 = self.__create_gin_convolution(hidden_features, hidden_features)

        self.batch_norm_1 = torch.nn.BatchNorm1d(hidden_features)
        self.batch_norm_2 = torch.nn.BatchNorm1d(hidden_features)
        self.batch_norm_3 = torch.nn.BatchNorm1d(hidden_features)
        self.batch_norm_4 = torch.nn.BatchNorm1d(hidden_features)
        self.batch_norm_5 = torch.nn.BatchNorm1d(hidden_features)

        self.fully_connected_1 = torch.nn.Linear(hidden_features, hidden_features)
        self.fully_connected_2 = torch.nn.Linear(hidden_features, out_features)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def __create_gin_convolution(self, in_features: int, out_features: int) -> geometric.nn.GINConv:
        return geometric.nn.GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, out_features))
        )

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data["graph"]
        x = data.x.float()

        x = self.activation(self.convolution_1(x, data.edge_index))
        x = self.batch_norm_1(x)

        x = self.activation(self.convolution_2(x, data.edge_index))
        x = self.batch_norm_2(x)

        x = self.activation(self.convolution_3(x, data.edge_index))
        x = self.batch_norm_3(x)

        x = self.activation(self.convolution_4(x, data.edge_index))
        x = self.batch_norm_4(x)

        x = self.activation(self.convolution_5(x, data.edge_index))
        x = self.batch_norm_5(x)

        x = geometric.nn.global_add_pool(x, data.batch)
        x = self.fully_connected_1(x)
        x = self.activation(x)

        x = self.dropout(x)
        x = self.fully_connected_2(x)

        return x
