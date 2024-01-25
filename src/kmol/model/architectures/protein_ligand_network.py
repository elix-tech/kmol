from typing import Dict, Any, List

import torch

from kmol.model.layers import LinearBlock, MultiplicativeInteractionLayer
from kmol.model.architectures.abstract_network import AbstractNetwork


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
        self.out_features = out_features
        self.protein_module = protein_module
        self.ligand_module = ligand_module

        self.output_module = LinearBlock(
            ligand_module.out_features + protein_module.out_features, hidden_features, out_features
        )

        if xavier_init:
            self.protein_module.apply(self._init_weights)
            self.ligand_module.apply(self._init_weights)
            self.output_module.apply(self._init_weights)
        self.last_hidden_layer_name = "output_module.block.1"
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
