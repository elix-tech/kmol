from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Optional, List, Union

import torch
import torch_geometric as geometric

from .layers import GraphConvolutionWrapper, TripletMessagePassingLayer, LinearBlock, MultiplicativeInteractionLayer
from ..core.helpers import Namespace, SuperFactory
from ..core.logger import LOGGER as logging
from ..core.observers import EventManager
from ..core.exceptions import CheckpointNotFound
from ..model.read_out import get_read_out


class AbstractNetwork(torch.nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def get_requirements(self) -> List[str]:
        raise NotImplementedError

    def map(self, module: "AbstractNetwork", *args) -> Dict[str, Any]:
        requirements = module.get_requirements()

        if len(args) != len(requirements):
            raise AttributeError("Cannot map inputs to module")

        return {requirement: args[index] for index, requirement in enumerate(requirements)}

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if checkpoint_path is None:
            raise CheckpointNotFound

        if device is None:
            device = torch.device("cpu")

        logging.info("Restoring from Checkpoint: {}".format(checkpoint_path))
        info = torch.load(checkpoint_path, map_location=device)

        payload = Namespace(network=self, info=info)
        EventManager.dispatch_event(event_name="before_checkpoint_load", payload=payload)

        self.load_state_dict(info["model"], strict=False)

    @staticmethod
    def dropout_layer_switch(m, dropout_prob):
        if isinstance(m, torch.nn.Dropout):
            if dropout_prob is not None:
                m.p = dropout_prob
            m.train()

    def activate_dropout(self, dropout_prob):
        self.apply(lambda m: self.dropout_layer_switch(m, dropout_prob))

    def mc_dropout(self, data, dropout_prob=None, n_iter=5):
        self.activate_dropout(dropout_prob)

        outputs = torch.stack([self.forward(data) for _ in range(n_iter)], dim=0)
        return {
            "logits": torch.mean(outputs, dim=0),
            "logits_var": torch.var(outputs, dim=0)
        }


class EnsembleNetwork(AbstractNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__()
        self.models = torch.nn.ModuleList([SuperFactory.create(AbstractNetwork, config) for config in model_configs])

    def load_checkpoint(self, checkpoint_paths: List[str], device: Optional[torch.device] = None):
        n_models = len(self.models)
        n_checkpoints = len(checkpoint_paths)
        if n_models != n_checkpoints:
            raise ValueError(
                f"Number of checkpoint_path should be equal to number of models. Received {n_models}, {n_checkpoints}."
            )
        for model, checkpoint_path in zip(self.models, checkpoint_paths):
            model.load_checkpoint(checkpoint_path, device)

    def get_requirements(self):
        return list(set(sum([model.get_requirements() for model in self.models], [])))

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = torch.stack([model.forward(data) for model in self.models], dim=0)
        return {
            "logits": torch.mean(outputs, dim=0),
            "logits_var": torch.var(outputs, dim=0)
        }

    def mc_dropout(
            self,
            data,
            dropout_prob=None,
            n_iter=5,
            return_distrib=False,
    ):
        self.activate_dropout(dropout_prob)
        means, vars = zip(*[model.mc_dropout(data, dropout_prob, n_iter).values() for model in self.models])
        means = torch.stack(means, dim=0)
        mean = means.mean(dim=0)
        var = (torch.stack(vars, dim=0).mean(dim=0) + means.var(dim=0)) / 2

        return {"logits": mean, "logits_var": var}


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
            **kwargs
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
                **kwargs
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
                    **kwargs
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

        self.last_hidden_layer_name = 'mlp.1'

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
        self.last_hidden_layer_name = 'mlp.1'

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
        self.last_hidden_layer_name = 'mlp.2'

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


class LinearNetwork(AbstractNetwork, LinearBlock):
    def get_requirements(self) -> List[str]:
        return ["features"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        features = data[self.get_requirements()[0]]
        return super().forward(features)


class ConvolutionalNetwork(AbstractNetwork):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.convolutional_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_features, out_channels=10, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(16),
            torch.nn.ReLU(),
        )
        self.last_hidden_layer_name = 'linear_block.block.1'
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
            self,
            protein_module: AbstractNetwork,
            ligand_module: AbstractNetwork,
            hidden_features: int = 0,
            out_features: int = 0,
            use_mi: bool = False,
            xavier_init: bool = True,
    ):
        super().__init__()

        self.protein_module = protein_module
        self.ligand_module = ligand_module
        self.output_module = LinearBlock(
            ligand_module.out_features + protein_module.out_features, hidden_features, out_features)

        if xavier_init:
            self.protein_module.apply(self._init_weights)
            self.ligand_module.apply(self._init_weights)
            self.output_module.apply(self._init_weights)
        self.last_hidden_layer_name = 'output_module.block.1'
        self.activation = torch.nn.ReLU()

        self.use_mi = use_mi

        if use_mi:
            self.mi_layer = MultiplicativeInteractionLayer(
                input_dim=ligand_module.out_features,
                context_dim=protein_module.out_features,
                output_dim=protein_module.out_features + ligand_module.out_features,
            )
            if xavier_init:
                self.mi_layer.apply(self._init_weights)

    def _init_weights(self, layer: torch.nn) -> None:
        if type(layer) == torch.nn.Linear:
            layer.weight.data.copy_(torch.nn.init.xavier_uniform_(layer.weight.data))

    def get_requirements(self) -> List[str]:
        return ["ligand", "protein"]

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        requirements = self.get_requirements()

        ligand_features = self.map(self.ligand_module, data[requirements[0]])
        protein_features = self.map(self.protein_module, data[requirements[1]])

        protein_features = self.activation(self.protein_module(protein_features))
        ligand_features = self.activation(self.ligand_module(ligand_features))

        if not self.use_mi:
            combined_features = torch.cat((protein_features, ligand_features), dim=-1)
        else:
            combined_features = self.mi_layer(ligand_features, protein_features)

        output = self.output_module(combined_features)

        return output
