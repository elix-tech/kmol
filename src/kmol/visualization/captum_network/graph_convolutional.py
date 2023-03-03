from typing import Any, Dict, Optional, Union, List
from captum.attr._core.integrated_gradients import IntegratedGradients
import torch
from .abstract_captum_network import AbstractCaptumNetwork
from ...model.architectures import GraphConvolutionalNetwork


class GraphConvolutionalCaptumNetwork(AbstractCaptumNetwork, GraphConvolutionalNetwork):
    """
    Captum class for Contribution computation of the Graph convolutional Network.
    It will provide the contribution of the graph and molecule features.
    If the model only use a graph this class is not useful.
    """

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
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            molecule_features,
            dropout,
            layer_type,
            layers_count,
            concat_layers,
            is_residual,
            norm_layer,
            activation,
            molecule_hidden,
            read_out,
            read_out_kwargs,
            final_activation,
            **kwargs,
        )
        self.ig_outputs = ["ig_graph", "ig_descriptors"]

    def get_integrate_gradient(self, data: Dict[str, Any], **kwargs) -> torch.Tensor:

        read_out, molecule_feature = self.get_ig_input(data)
        model = self.get_integrate_model()

        read_out.requires_grad_()
        molecule_feature.requires_grad_()

        ig = IntegratedGradients(model)

        graph, molecule_feature = ig.attribute((read_out, molecule_feature), **kwargs)

        return {"graph_feature": graph.detach().cpu().numpy(), "mol_feature": molecule_feature.detach().cpu().numpy()}

    def get_integrate_model(self):
        class GraphConvolutionalCaptumModel(torch.nn.Module):
            def forward(_self, read_out, molecule_features):

                molecule_features = self.molecular_head(molecule_features)

                x = torch.cat((read_out, molecule_features), dim=1)
                x = self.mlp(x)
                return x

        return GraphConvolutionalCaptumModel()

    def get_ig_input(self, data):
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

        return read_out, data.molecule_features
