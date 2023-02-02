from typing import Dict, Any, Optional, List, Union

import torch

from ..layers import GraphConvolutionWrapper
from ...core.helpers import SuperFactory
from ...model.read_out import get_read_out


from .abstract_network import AbstractNetwork


class GraphConvolutionalNetwork(AbstractNetwork):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        molecule_features: int,
        dropout: float,
        layer_type: str = "torch_geometric.nn.GCNConv",
        layers_count: int = 2,
        concat_layers: bool = False,
        is_residual: bool = True,
        norm_layer: Optional[str] = None,
        activation: str = "torch.nn.ReLU",
        molecule_hidden: Optional[int] = None,
        read_out: Union[str, List[str]] = ("max", "sum"),
        read_out_kwargs: Optional[Dict[str, Any]] = None,
        final_activation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.out_features = out_features
        self.concat_layers = concat_layers
        self.convolutions = torch.nn.ModuleList()
        self.convolutions.append(
            GraphConvolutionWrapper(
                in_features=in_features,
                out_features=hidden_features,
                dropout=dropout,
                layer_type=layer_type,
                is_residual=is_residual,
                norm_layer=norm_layer,
                activation=activation,
                **kwargs,
            )
        )

        for _ in range(layers_count - 1):
            self.convolutions.append(
                GraphConvolutionWrapper(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    dropout=dropout,
                    layer_type=layer_type,
                    is_residual=is_residual,
                    norm_layer=norm_layer,
                    activation=activation,
                    **kwargs,
                )
            )
        if read_out_kwargs is None:
            read_out_kwargs = {}
        read_out_kwargs.update({"in_channels": hidden_features})
        self.read_out = get_read_out(read_out, read_out_kwargs)
        self.molecular_head = lambda x: torch.Tensor().to(x.device)
        molecule_hidden = hidden_features // 4 if molecule_hidden is None else molecule_hidden
        if molecule_features:
            self.molecular_head = torch.nn.Sequential(
                torch.nn.Linear(molecule_features, molecule_hidden),
                torch.nn.Dropout(p=min(hidden_features / in_features, 0.7)),
                torch.nn.BatchNorm1d(molecule_hidden),
                torch.nn.ReLU(),
            )
        readout_out_dim = self.read_out.out_dim * layers_count if self.concat_layers else self.read_out.out_dim
        mlp_features = readout_out_dim + bool(molecule_features) * molecule_hidden

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_features, out_features),
        )

        if final_activation is not None:
            self.mlp.add_module("final_activation", SuperFactory.reflect(final_activation)())

        self.last_hidden_layer_name = "mlp.1"

    def get_requirements(self) -> List[str]:
        return ["graph"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        data = data[self.get_requirements()[0]]
        x = data.x.float()

        layers_outputs = []
        for convolution in self.convolutions:
            x = convolution(x, data.edge_index, data.edge_attr, data.batch)
            layers_outputs.append(x)

        if self.concat_layers:
            read_out = torch.cat([self.read_out(e, batch=data.batch) for e in layers_outputs], dim=1)
        else:
            read_out = self.read_out(x, batch=data.batch)
        molecule_features = self.molecular_head(data.molecule_features)

        x = torch.cat((read_out, molecule_features), dim=1)
        x = self.mlp(x)

        return x
