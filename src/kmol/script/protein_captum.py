from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

from ..model.architectures import ProteinSchnetNetwork
from .integrated_gradient import CaptumScript, GRAPH_CAPTUM_FEAT

PROTEIN_FEATURES = ["protein", "schnet_input"]
GRAPH_CAPTUM_FEAT += ["z", "z_protein", "coords", "edge_attr"]

MODEL_ATTRIBUTION = {
    ProteinSchnetNetwork: {
        "type": "captum.attr.LayerIntegratedGradients",
        "layer": ["embedding.0", "protein_embedding.0", "protein_embedding.1", "interactions.0.mlp_protein"],
    }
}


class ProteinCaptumScript(CaptumScript):
    def __init__(self, config_path, protein_input_name, processing_type, n_steps: int = 50) -> None:
        super().__init__({}, config_path, "sum", n_steps)
        self.attribution, groups = self.get_integrated_layer_protein_schnet(self.model)
        self.attribution_innit()
        self.protein_input_name = protein_input_name
        self._process = getattr(self, f"process_{processing_type}")
        self.groups = groups

    def run(self):
        protein_inputs = self.dataset[self.protein_input_name].values.tolist()
        for protein_raw_data, data in tqdm(zip(protein_inputs, self.data_loader)):
            self.tmp_executor._to_device(data)
            attributions = []
            attribute_result = self.attributor.attribute(data.inputs, target=0, n_steps=self.n_steps)
            results = []
            for i in range(len(self.groups) - 1):
                result = torch.concat(attribute_result[self.groups[i] : self.groups[i + 1]], axis=1).sum(axis=1)
                results.append(result.detach().cpu().numpy())
            ligand_atom, protein_atom, lp_interaction = results
            atoms = np.concatenate([ligand_atom, protein_atom])
            edge_index = data.inputs["schnet_inputs"].edge_index.detach().cpu().numpy()
            for i in range(len(lp_interaction)):
                src, dest = edge_index[0, i], edge_index[1, i]
                atoms[src] = atoms[src] + lp_interaction[i] / 2
                atoms[dest] = atoms[dest] + lp_interaction[i] / 2
            # TODO Add conversion to original atoms and add to files.
            # attributions.append(attribution[0].detach().cpu().numpy())

            # target_sequence case
            self._process(protein_raw_data, attributions)

    def process_target_sequence(self, protein_raw_data, attributions):
        result = ""
        attributions = [a.squeeze() for a in attributions]
        for data in enumerate(zip(protein_raw_data, *attributions)):
            pass

    def get_integrated_layer_protein_schnet(self, model: ProteinSchnetNetwork):
        ligand_embedding = [f"embedding.{i}" for i in range(len(model.embedding))]
        protein_embedding = [f"protein_embedding.{i}" for i in range(len(model.protein_embedding))]
        interaction_feature = ["interactions.0.mlp_protein"]
        groups = np.cumsum([0, len(ligand_embedding), len(protein_embedding), 1])
        return {
            "type": "captum.attr.LayerIntegratedGradients",
            "layer": ligand_embedding + protein_embedding + interaction_feature,
        }, groups
