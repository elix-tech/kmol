from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from openbabel import pybel
import pandas as pd

from kmol.model.architectures import ProteinSchnetNetwork
from kmol.script.integrated_gradient import CaptumScript, GRAPH_CAPTUM_FEAT

PROTEIN_FEATURES = ["protein", "schnet_input"]
GRAPH_CAPTUM_FEAT += ["z", "z_protein", "coords"]


class ProteinCaptumScript(CaptumScript):
    def __init__(self, config_path, n_steps: int = 50, default_attribution_value: int = 0) -> None:
        super().__init__({}, config_path, "sum", n_steps)
        self.attribution = self.get_attribution(self.model)
        self.attribution_innit()
        self.default_attribution_value = default_attribution_value

    def run(self):
        for data in tqdm(self.data_loader):
            self.tmp_executor._to_device(data)
            target = self.model(data.inputs).argmax(axis=1).detach().cpu().item()
            attribute_result = self.attributor.attribute(data.inputs, target=target, n_steps=self.n_steps)
            self.process(data, attribute_result)
        self.post_process()

    def get_attribution(self, model: ProteinSchnetNetwork):
        ligand_embedding = [f"embedding.{i}" for i in range(len(model.embedding))]
        protein_embedding = [f"protein_embedding.{i}" for i in range(len(model.protein_embedding))]
        interaction_feature = ["interactions.0.mlp_protein"]
        self.groups = np.cumsum([0, len(ligand_embedding), len(protein_embedding), 1])
        return {
            "type": "captum.attr.LayerIntegratedGradients",
            "layer": ligand_embedding + protein_embedding + interaction_feature,
        }

    def process(self, data, attribute_result):
        attributions = []
        results = []
        for i in range(len(self.groups) - 1):
            result = torch.concat(attribute_result[self.groups[i] : self.groups[i + 1]], axis=1).sum(axis=1)
            results.append(result.detach().cpu().numpy())
        ligand_atom, protein_atom, lp_interaction = results
        attributions = np.concatenate([ligand_atom, protein_atom])
        edge_index = data.inputs["schnet_inputs"].edge_index.detach().cpu().numpy()
        for i in range(len(lp_interaction)):
            src, dest = edge_index[0, i], edge_index[1, i]
            attributions[src] = attributions[src] + lp_interaction[i] / 2
            attributions[dest] = attributions[dest] + lp_interaction[i] / 2

        self.generate_pdb(
            mol2_path=data.inputs["schnet_inputs"].mol2_path[0],
            attributions=attributions,
            orinal_ids_mapping=data.inputs["schnet_inputs"].original_atom_ids.detach().cpu().numpy(),
        )

    def post_process(self):
        pass

    def generate_pdb(self, mol2_path, attributions, orinal_ids_mapping):
        mol = next(pybel.readfile("mol2", mol2_path))
        pdb_output = mol.write("pdb").splitlines()

        # Modify PDB lines with new B-factor values
        modified_pdb_output = []
        for line in pdb_output:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_idx = int(line[6:11].strip()) - 1
                mask = orinal_ids_mapping == atom_idx
                if mask.any():
                    line = line[:60] + f"{attributions[mask].item():6.2f}" + line[66:]
                else:
                    line = line[:60] + f"{self.default_attribution_value:6.2f}" + line[66:]

            modified_pdb_output.append(line)

        save_dir = Path(self._config.output_path) / "protein_captum"
        save_dir.mkdir(exist_ok=True, parents=True)
        pdb_file = save_dir / Path(mol2_path).with_suffix(".attribution.pdb").name
        with open(pdb_file, "w") as f:
            f.write("\n".join(modified_pdb_output))


class ProteinSequenceCaptumScript(ProteinCaptumScript):
    def __init__(
        self,
        config_path: str,
        sequence_column: str,
        smiles_column: str = "smiles",
        n_steps: int = 50,
        default_attribution_value: int = 0,
    ):
        super().__init__(config_path, n_steps, default_attribution_value)
        self.sequence_column = sequence_column
        self.smiles_column = smiles_column
        self.sequence = self.dataset[self.sequence_column]
        self.smiles = self.dataset[self.smiles_column]
        self.results = pd.DataFrame(columns=[self.sequence_column, smiles_column, "path_attribution"])

    def process(self, data, attribute_result):
        # Overall output file update
        protein_attribution = dict(zip(self.attributor.output_name, attribute_result))["protein"].detach().cpu().numpy()
        dataset_info = self.dataset.iloc[len(self.results)][[self.sequence_column, self.smiles_column]].values
        path_attribution = Path(self._config.output_path) / "protein_attribution" / f"atribution_{len(self.results)}.csv"
        self.results.loc[len(self.results)] = np.concatenate([dataset_info, [str(path_attribution)]])

        # Attribution file creation
        path_attribution.parent.mkdir(parents=True, exist_ok=True)
        attribution_sequence = pd.DataFrame(columns=["residue", "attribution"])
        size_sequence = protein_attribution.shape[-1]
        sequence = dataset_info[0]
        for i, residue in enumerate(sequence):
            if i >= size_sequence:
                score = 0
            else:
                score = protein_attribution[:, i].item()
            attribution_sequence.loc[len(attribution_sequence)] = [residue, score]
        attribution_sequence.to_csv(path_attribution, index=False)

    def post_process(self):
        path_results = Path(self._config.output_path) / "attribution_file_index.csv"
        self.results.to_csv(path_results, index=False)

    def get_attribution(self, *args, **kwargs):
        return {"type": "captum.attr.IntegratedGradients"}
