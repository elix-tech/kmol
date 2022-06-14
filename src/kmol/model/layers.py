from typing import Optional, Dict

import torch
import torch_geometric
from torch.nn.functional import leaky_relu
from torch_scatter import scatter_mean, scatter_std, scatter

from ..core.helpers import SuperFactory


class GraphConvolutionWrapper(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
            layer_type: str = "torch_geometric.nn.GCNConv",
            is_residual: bool = True,
            norm_layer: Optional[str] = None,
            activation: str = "torch.nn.ReLU",
            edge_features: int = 0,
            propagate_edge_features: bool = False,
            **kwargs
    ):
        super().__init__()
        base_features = in_features + in_features // 2 if edge_features else in_features
        self.convolution = SuperFactory.reflect(layer_type)(base_features, out_features, **kwargs)

        self._propagate_edge_features = propagate_edge_features
        self._edge_features = edge_features
        if self._edge_features and not self._propagate_edge_features:
            self.edge_projection = torch.nn.Linear(edge_features, in_features // 2)

        self.norm_layer = SuperFactory.reflect(norm_layer)(out_features) if norm_layer else None
        self.residual_layer = torch.nn.Linear(base_features, out_features) if is_residual else None
        self.activation = SuperFactory.reflect(activation)()
        self.dropout = torch.nn.Dropout(p=dropout)

    def _get_layer_arguments(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        arguments = {"x": x, "edge_index": edge_index}
        if self._propagate_edge_features:
            arguments["edge_attr"] = edge_attr

        return arguments

    def _add_edge_features(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:

        if self._edge_features and not self._propagate_edge_features:
            last_atom_index = x.size(0) - 1

            if last_atom_index not in torch.unique(edge_index[0]):  # fix issue when last node has no bond
                edge_index = torch.cat(
                    (
                        edge_index,
                        torch.LongTensor([[last_atom_index], [last_atom_index]]).to(edge_index.device),
                    ),
                    dim=1,
                )

                edge_attr = torch.cat(
                    (
                        edge_attr,
                        torch.zeros((1, edge_attr.size(1))).to(edge_attr.device),
                    ),
                    dim=0,
                )

            per_node_edge_features = scatter(edge_attr, edge_index[0], dim=0, reduce="sum")
            per_node_edge_features = leaky_relu(self.edge_projection(per_node_edge_features))

            x = torch.cat([x, per_node_edge_features], dim=1)

        return x

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch: torch.Tensor,
    ) -> torch.Tensor:

        x = self._add_edge_features(x, edge_index, edge_attr)
        identity = x

        arguments = self._get_layer_arguments(x, edge_index, edge_attr)
        x = self.convolution(**arguments)

        if self.residual_layer:
            x += self.residual_layer(identity)

        if self.norm_layer:
            x = self.norm_layer(x, batch)

        x = self.activation(x)
        x = self.dropout(x)

        return x


class LinearBlock(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            activation: str = "torch.nn.ReLU",
            dropout: float = 0.,
            use_batch_norm: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        layers = [
            torch.nn.Linear(in_features, hidden_features),
        ]
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_features))
        layers.append(SuperFactory.reflect(activation)())
        if dropout:
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(hidden_features, out_features))
        self.block = torch.nn.Sequential(*layers)
        self.last_hidden_layer = f'block.{len(layers)-1}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GINConvolution(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.convolution = torch_geometric.nn.GINConv(
            LinearBlock(
                in_features=in_features,
                hidden_features=out_features,
                out_features=out_features,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.convolution(x, edge_index)


class TrimConvolution(torch_geometric.nn.MessagePassing):
    """
    Graph convolution as introduced in https://doi.org/10.1093/bib/bbaa266.
    Implementation taken from https://github.com/yvquanli/TrimNet.
    """

    def __init__(self, in_features, out_features, in_edge_features, heads=4, negative_slope=0.2, **kwargs):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        # node_dim = 0 for multi-head aggr support
        self.in_features = in_features
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = torch.nn.Parameter(torch.Tensor(in_features, heads * in_features))
        self.weight_edge = torch.nn.Parameter(torch.Tensor(in_edge_features, heads * in_features))
        self.weight_triplet_att = torch.nn.Parameter(torch.Tensor(1, heads, 3 * in_features))
        self.weight_scale = torch.nn.Parameter(torch.Tensor(heads * in_features, out_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_node)
        torch.nn.init.kaiming_uniform_(self.weight_edge)
        torch.nn.init.kaiming_uniform_(self.weight_triplet_att)
        torch.nn.init.kaiming_uniform_(self.weight_scale)
        torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size=None,
    ) -> torch.Tensor:
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.in_features)
        x_i = x_i.view(-1, self.heads, self.in_features)
        e_ij = edge_attr.view(-1, self.heads, self.in_features)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        alpha = torch.nn.functional.leaky_relu(alpha, self.negative_slope)
        alpha = torch_geometric.utils.softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)

        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.in_features)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias

        return aggr_out

    def extra_repr(self):
        return "{node_channels}, {node_channels}, heads={heads}".format(**self.__dict__)


class TripletMessagePassingLayer(torch.nn.Module):
    """
    Message passing layer as introduced in https://doi.org/10.1093/bib/bbaa266.
    Implementation taken from https://github.com/yvquanli/TrimNet.
    """

    def __init__(self, node_channels, edge_channels: int, heads: int = 4, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.convolution = TrimConvolution(node_channels, node_channels, edge_channels, heads)

        self.gru = torch.nn.GRU(node_channels, node_channels)
        self.layer_norm = torch.nn.LayerNorm(node_channels)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)

        for _ in range(self.steps):
            m = self.convolution.forward(x, edge_index, edge_attr)
            m = torch.nn.functional.celu(m)

            x, h = self.gru(m.unsqueeze(0), h)
            x = self.layer_norm(x.squeeze(0))

        return x


class GraphNorm(torch.nn.Module):
    """
    Normalization layer introduced in https://arxiv.org/abs/2009.03294
    Implementation is based on: https://github.com/lsj2408/GraphNorm
    """

    def __init__(self, hidden_dimension: int):
        super().__init__()

        self.epsilon = 1e-8

        self.weight = torch.nn.Parameter(torch.Tensor(hidden_dimension))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_dimension))

        self.mean_scale = torch.nn.Parameter(torch.Tensor(hidden_dimension))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.ones_(self.mean_scale)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):

        mean = scatter_mean(x, batch, dim=0)
        std = scatter_std(x, batch, dim=0)
        std = torch.add(std, self.epsilon)

        x = x - mean[batch] * self.mean_scale
        out = self.weight * x / std[batch] + self.bias

        return out


class BatchNorm(torch_geometric.nn.BatchNorm):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class MultiplicativeInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim: int, context_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        self.mi_lin1 = torch.nn.Linear(context_dim, output_dim * input_dim)
        self.mi_lin2 = torch.nn.Linear(context_dim, output_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        weights = self.mi_lin1(z)
        bias = self.mi_lin2(z)

        weights = weights.view(-1, self.input_dim, self.output_dim)
        out = torch.bmm(x.unsqueeze(1), weights).squeeze(1) + bias
        return out
