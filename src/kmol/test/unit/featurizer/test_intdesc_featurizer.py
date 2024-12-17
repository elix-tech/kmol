from typing import List
from tempfile import TemporaryDirectory
from pathlib import Path
import pickle

import torch
import torch_geometric

from kmol.core.helpers import SuperFactory
from kmol.data.featurizers import AbstractFeaturizer
from kmol.data.resources import DataPoint


class TestSchnetFeaturizationPipeline:
    data_location = Path("src/kmol/test/data/featurizer/intdesc_featurizer/")

    PDB_DIR = TemporaryDirectory()
    ANTECHAMBER = TemporaryDirectory()
    INTDESC = TemporaryDirectory()

    SAMPLES = ["G3V9H8-drug_id-64833", "P35968-drug_id-63051", "Q9P1W9-drug_id-467400"]

    SHARED_FEATURIZER_CONFIG = [
        {
            "type": "pdbqt_to_pdb",
            "inputs": ["pdbqt_filepath"],
            "outputs": ["path_pdb"],
            "pdqt_dir": data_location,
            "dir_to_save": PDB_DIR.name,
            "protonize": False,
        },
        {
            "type": "intdesc",
            "inputs": ["path_pdb"],
            "outputs": ["schnet_inputs"],
            "ligand_residue": ["UNL"],
            "dir_to_save": INTDESC.name,
            "dup": True,
        },
    ]

    def run_test_sample(self, name_sample: str, featurizers: List[AbstractFeaturizer]):
        example_data = DataPoint(0, [], {"pdbqt_filepath": f"{name_sample}.pdbqt"}, [])

        for featurizer in featurizers:
            featurizer.run(example_data)

        return example_data

    def run_pipeline(self, featurizers: List[AbstractFeaturizer], data_dir: str):
        torch_geometric.seed_everything(42)
        for sample in self.SAMPLES:
            output = self.run_test_sample(sample, featurizers)
            self.validate_results(data_dir, sample, output)

    def get_featurizers(self, featurizer_config):
        return [SuperFactory.create(AbstractFeaturizer, f) for f in featurizer_config["featurizers"]]

    def validate_results(self, data_dir, name_sample, output):
        schnet_input = output.inputs["schnet_inputs"]
        with open(Path(data_dir) / f"schnet_input_{name_sample}.pkl", "rb") as handle:
            valid_input = pickle.load(handle)

        assert (valid_input["edge_index"] == schnet_input["edge_index"]).all()
        assert (valid_input["edge_attr"] == schnet_input["edge_attr"]).all()
        assert (valid_input["z"] == schnet_input["z"]).all()
        assert (valid_input["z_protein"] == schnet_input["z_protein"]).all()
        assert torch.allclose(valid_input["coords"], schnet_input["coords"])
        assert (valid_input["protein_mask"] == schnet_input["protein_mask"]).all()
        assert (valid_input["original_atom_ids"] == schnet_input["original_atom_ids"]).all()

    def test_ligand_sybyl(self):
        featurizer_config = {
            "featurizers": self.SHARED_FEATURIZER_CONFIG
            + [
                {
                    "type": "atom_type_extension",
                    "inputs": ["schnet_inputs"],
                    "outputs": ["schnet_inputs"],
                    "ligand_residue": ["UNL"],
                    "protein_atom_type": ["SYBYL", "PDB"],
                    "ligand_atom_type": ["SYBYL"],
                    "dir_to_save": self.ANTECHAMBER.name,
                    "rewrite": False,
                },
            ]
        }

        featurizers = self.get_featurizers(featurizer_config)
        self.run_pipeline(featurizers, data_dir=self.data_location / "ligand_sybyl")

    def test_ligand_sybyl_bcc(self):
        featurizer_config = {
            "featurizers": self.SHARED_FEATURIZER_CONFIG
            + [
                {
                    "type": "atom_type_extension",
                    "inputs": ["schnet_inputs"],
                    "outputs": ["schnet_inputs"],
                    "ligand_residue": ["UNL"],
                    "protein_atom_type": ["SYBYL", "PDB"],
                    "ligand_atom_type": ["SYBYL", "AM1-BCC"],
                    "dir_to_save": self.ANTECHAMBER.name,
                    "rewrite": False,
                },
            ]
        }

        featurizers = self.get_featurizers(featurizer_config)
        self.run_pipeline(featurizers, data_dir=self.data_location / "ligand_sybyl_bcc")

    def test_ligand_gaff_sybyl(self):
        featurizer_config = {
            "featurizers": self.SHARED_FEATURIZER_CONFIG
            + [
                {
                    "type": "atom_type_extension",
                    "inputs": ["schnet_inputs"],
                    "outputs": ["schnet_inputs"],
                    "ligand_residue": ["UNL"],
                    "protein_atom_type": ["SYBYL", "PDB"],
                    "ligand_atom_type": ["GAFF", "SYBYL"],
                    "dir_to_save": self.ANTECHAMBER.name,
                    "rewrite": False,
                },
            ]
        }

        featurizers = self.get_featurizers(featurizer_config)
        self.run_pipeline(featurizers, data_dir=self.data_location / "ligand_gaff_sybyl")

    def test_protein_pdb_sybyl(self):

        featurizer_config = {
            "featurizers": self.SHARED_FEATURIZER_CONFIG
            + [
                {
                    "type": "atom_type_extension",
                    "inputs": ["schnet_inputs"],
                    "outputs": ["schnet_inputs"],
                    "ligand_residue": ["UNL"],
                    "protein_atom_type": ["PDB", "SYBYL"],
                    "ligand_atom_type": ["SYBYL"],
                    "dir_to_save": self.ANTECHAMBER.name,
                    "rewrite": False,
                },
            ]
        }

        featurizers = self.get_featurizers(featurizer_config)
        self.run_pipeline(featurizers, data_dir=self.data_location / "protein_pdb_sybyl")
