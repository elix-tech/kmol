from copy import deepcopy
from typing import Any, Dict
from captum.attr._core.integrated_gradients import IntegratedGradients
import torch
import numpy as np

from .abstract_captum_network import AbstractCaptumNetwork
from ...model.architectures import AbstractNetwork, ProteinLigandNetwork


class ProteinLigandCaptumNetwork(AbstractCaptumNetwork, ProteinLigandNetwork):
    """
    Model for Captum contribution computation. Link with the ProteinLigand Network
    """

    def __init__(
        self,
        protein_module: AbstractNetwork,
        ligand_module: AbstractCaptumNetwork,
        hidden_features: int = 0,
        out_features: int = 0,
        use_mi: bool = False,
        xavier_init: bool = True,
    ):

        super().__init__(protein_module, ligand_module, hidden_features, out_features, use_mi, xavier_init)
        self.ig_outputs = ["ig_protein", "ig_graph", "ig_descriptors"]
        # self.ligand_module = SuperFactory.create(AbstractCaptumNetwork, ligand_module)
        self.out_features = out_features

    def get_integrate_gradient(self, data: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Retrieve the ig of the feature of interest. For this model, ie protein graph and molecule.
        """

        protein_features, read_out, molecule_feature, ligand_feature = self.get_ig_input(deepcopy(data))

        protein_features.requires_grad_()
        read_out.requires_grad_()
        molecule_feature.requires_grad_()

        # Captum processing
        detail_model = self.get_integrate_model()
        ig_intermediate = IntegratedGradients(deepcopy(detail_model.base_model), multiply_by_inputs=False)

        features = []
        for i in range(self.out_features):
            protein, ligand = ig_intermediate.attribute(
                (protein_features, ligand_feature),
                target=i,
                **kwargs,
            )

            ig_detail = IntegratedGradients(detail_model, multiply_by_inputs=False)
            graph, descriptors = ig_detail.attribute(
                (read_out, molecule_feature),
                target=i,
                additional_forward_args=protein_features,
                **kwargs,
            )
            # TODO Compute the concatenation of result in case of multiple output
            features.append(self.compute_sum(ligand, protein, graph, descriptors))

        feature_sum = {k: np.array([]) for k in self.ig_outputs}
        for dict_ in features:
            for i, (k, v) in enumerate(dict_.items()):
                feature_sum[self.ig_outputs[i]] = np.append(feature_sum[self.ig_outputs[i]], v)

        # topk = self.compute_top_k_percent(ligand, protein, graph, descriptors)

        return feature_sum  # , topk

    def compute_sum(self, ligand, protein, graph, descriptors):
        """
        Compute the sum of each feature of interest and rescale them as percentage.
        """

        total_contribution = ligand.abs().sum() + protein.abs().sum()
        ligand = ligand.abs().sum() / total_contribution
        protein = protein.abs().sum() / total_contribution

        total_contribution = graph.abs().sum() + descriptors.abs().sum()
        descriptors = descriptors.abs().sum() / total_contribution * ligand
        graph = graph.abs().sum() / total_contribution * ligand

        return {
            "protein": protein.detach().cpu().numpy(),
            "graph": graph.detach().cpu().numpy(),
            "descriptors": descriptors.detach().cpu().numpy(),
        }

    def compute_top_k_percent(self, ligand, protein, graph, descriptors, k=10):
        """
        Sum the 10% higher contribution of each feature of interest.
        """
        ligand = torch.topk(ligand.abs(), int(k * len(ligand)))[0]
        protein = torch.topk(protein.abs(), int(k * len(protein)))[0]
        graph = torch.topk(graph.abs(), int(k * len(graph)))[0]
        descriptors = torch.topk(descriptors.abs(), int(k * len(descriptors)))[0]
        return self.compute_sum(ligand, protein, graph, descriptors)

    def get_integrate_model(self):
        """
        Generate a Pytorch model which take the feature of interest as input.
        Here it is compose of half of the model since our interest focus on
        intermediate feature.
        """

        class ProteinLigandCaptumModel(torch.nn.Module):
            def forward(_self, protein_features, ligand_features):
                if not self.use_mi:
                    combined_features = torch.cat((protein_features, ligand_features), dim=-1)
                else:
                    combined_features = self.mi_layer(ligand_features, protein_features)

                return self.output_module(combined_features)

        class DetailCaptumMode(torch.nn.Module):
            def __init__(_self):
                super().__init__()
                _self.ligand_module_captum = self.ligand_module.get_integrate_model()
                _self.base_model = ProteinLigandCaptumModel()

            def forward(_self, read_out, molecule_features, protein_features):
                # Apply Integrated gradiant model of the ligand module first
                ligand_features = self.activation(_self.ligand_module_captum(read_out, molecule_features))
                return _self.base_model(protein_features, ligand_features)

        return DetailCaptumMode()

    def get_ig_input(self, data: Dict[str, Any]):
        """
        Retrieve our feature of interest.
        Since the model use by Captum needs the feature of interest by input we
        need to pass our input in the first layers of the standard model.
        """
        # Regular forward pass until concatenation
        requirements = self.get_requirements()
        self.protein_module.eval()
        self.ligand_module.eval()
        ligand_features = self.map(self.ligand_module, data[requirements[0]])
        protein_features = self.map(self.protein_module, data[requirements[1]])

        protein_features = self.activation(self.protein_module(protein_features))
        # Retrieving meaningful feature of graph conv to get their integrated gradient
        read_out, molecule_feature = self.ligand_module.get_ig_input(ligand_features)

        ligand_features = self.activation(self.ligand_module(ligand_features))
        return protein_features, read_out, molecule_feature, ligand_features
