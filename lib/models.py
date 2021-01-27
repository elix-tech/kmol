import torch
import torch_geometric as geometric
from torch_geometric.data import Batch

from lib.resources import ProteinLigandBatch


class GraphConvolutionalNetwork(torch.nn.Module):

    def __init__(
            self, in_features: int, hidden_features: int, out_features:
            int, dropout: float, layer_type: str = "GraphConv", layers_count: int = 2
    ):
        super().__init__()
        convolution = getattr(geometric.nn, layer_type)

        self.convolutions = torch.nn.ModuleList()
        self.convolutions.append(convolution(in_features, hidden_features))

        for _ in range(layers_count - 2):
            self.convolutions.append(convolution(hidden_features, hidden_features))

        self.convolutions.append(convolution(hidden_features, out_features))

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data: Batch) -> torch.Tensor:

        x = data.x.float()

        for convolution in self.convolutions[:-1]:
            x = convolution(x, data.edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.convolutions[-1](x, data.edge_index)
        x = geometric.nn.global_max_pool(x, batch=data.batch)

        return x


class GraphIsomorphismNetwork(torch.nn.Module):

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

    def forward(self, data: Batch) -> torch.Tensor:

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


class BimodalProteinLigandNetwork(torch.nn.Module):

    def __init__(self, in_features_protein: int, in_features_ligand: int, out_features: int):
        super().__init__()

        self.protein_module = torch.nn.Sequential(
            torch.nn.Linear(in_features_protein, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16, bias=True),
            torch.nn.ReLU(),
        )

        self.ligand_module = torch.nn.Sequential(
            torch.nn.Linear(in_features_ligand, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 16, bias=True),
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

    def _init_weights(self, layer: torch.nn) -> None:
        if type(layer) == torch.nn.Linear:
            layer.weight.data.copy_(
                torch.nn.init.xavier_uniform_(layer.weight.data)
            )

    def forward(self, data: ProteinLigandBatch) -> torch.Tensor:
        protein_features = self.protein_module(data.protein_features)
        ligand_features = self.ligand_module(data.ligand_features)

        combined_features = torch.cat((protein_features, ligand_features), dim=-1)
        output = self.output_module(combined_features)

        return output
