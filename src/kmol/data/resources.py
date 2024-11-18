from abc import abstractmethod
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional, Iterable
from io import StringIO
from contextlib import redirect_stdout
import yaml
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater as TorchGeometricCollater
from kmol.vendor.graphormer import collater
import prody
import pandas as pd
from openbabel import openbabel

from kmol.vendor.riken.intDesc.interaction_descriptor import calculate

prody.LOGGER._setverbosity("error")


@dataclass
class DataPoint:
    id_: Optional[Union[str, int]] = None
    labels: Optional[List[str]] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Union[List[Any], np.ndarray]] = None


@dataclass
class Batch:
    ids: List[Union[str, int]]
    labels: List[str]
    inputs: Dict[str, torch.Tensor]
    outputs: torch.FloatTensor


@dataclass
class LoadedContent:
    dataset: Iterable[Batch]
    samples: int
    batches: int


class AbstractCollater:
    @abstractmethod
    def apply(self, batch: List[DataPoint]) -> Any:
        raise NotImplementedError


class GeneralCollater(AbstractCollater):
    def __init__(self):
        self._collater = TorchGeometricCollater(follow_batch=[], exclude_keys=[])

    def _unpack(self, batch: List[DataPoint]) -> Batch:
        ids = []
        inputs = defaultdict(list)
        outputs = []

        for entry in batch:
            ids.append(entry.id_)

            for key, value in entry.inputs.items():
                inputs[key].append(value)

            outputs.append(entry.outputs)

        outputs = torch.from_numpy(np.array(outputs))
        if outputs.dtype == torch.float64:
            outputs = outputs.type(torch.float32)
        inputs = dict(inputs)
        return Batch(ids=ids, labels=batch[0].labels, inputs=inputs, outputs=outputs)

    def apply(self, batch: List[DataPoint]) -> Batch:
        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            batch.inputs[key] = self._collater.collate(values)

        return batch


class GraphormerCollater(GeneralCollater):
    def __init__(self, max_node: int = 512, multi_hop_max_dist: int = 20, spatial_pos_max: int = 20):
        # TODO: automate increment of max_node without requiring pre-setting
        super().__init__()
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def apply(self, batch: List[DataPoint]) -> Batch:
        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            if key == "ligand":
                batch.inputs[key] = collater.collater(values, self.max_node, self.multi_hop_max_dist, self.spatial_pos_max)
            else:
                batch_dict = self._collater.collate(values)
                batch.inputs[key] = batch_dict

        return batch


class PaddedCollater(GeneralCollater):
    def __init__(self, padded_column):
        super().__init__()
        self.padded_column = padded_column

    def _pad_seq(self, seqs, dtype=torch.long):
        max_length = max([len(seq) for seq in seqs])
        # padding value is -1
        padded_seqs = [-1 * torch.ones(max_length, dtype=dtype) for _ in range(len(seqs))]
        for i, seq in enumerate(seqs):
            seq_tensor = seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=dtype)
            padded_seqs[i][: len(seq)] = seq_tensor.clone().detach()

        return padded_seqs

    def _unpack(self, batch: List[DataPoint]) -> Batch:
        ids = []
        inputs = defaultdict(list)
        outputs = []

        for entry in batch:
            ids.append(entry.id_)

            for key, value in entry.inputs.items():
                inputs[key].append(value)

            outputs.append(entry.outputs)

        inputs_padded = defaultdict(list)

        for key, values in inputs.items():
            if key == self.padded_column:
                inputs_padded[key] = self._pad_seq(values)
            else:
                inputs_padded[key] = values

        outputs_padded = self._pad_seq(outputs, dtype=torch.float)
        outputs_padded = torch.stack(outputs_padded)

        inputs_padded = dict(inputs_padded)

        return Batch(ids=ids, labels=batch[0].labels, inputs=inputs_padded, outputs=outputs_padded)


class IntDescRunner:
    """
    A class to run interaction descriptor calculations using the RIKEN intDesc tool.

    Attributes:
        int_desc_location (str): Path to the intDesc tool.
        interaction_labels (List[str]): List of interaction labels.
        intdesc_params (Dict[str, Any]): Parameters for the intDesc calculation.
    """

    int_desc_location = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/"))
    # fmt: off
    interaction_labels = [
        "LP_CH_F","LP_CH_Hal_Br","LP_CH_Hal_Cl","LP_CH_Hal_I","LP_CH_N","LP_CH_O","LP_CH_PI","LP_CH_S","LP_Ca_X","LP_Cl_X","LP_Dipo","LP_Elec_NH_N","LP_Elec_NH_O","LP_Elec_OH_N","LP_Elec_OH_O","LP_Fe_X","LP_HB_NH_N","LP_HB_NH_O","LP_HB_OH_N","LP_HB_OH_O","LP_Hal_Br_N","LP_Hal_Br_O","LP_Hal_Br_S","LP_Hal_Cl_N","LP_Hal_Cl_O","LP_Hal_Cl_S","LP_Hal_I_N","LP_Hal_I_O","LP_Hal_I_S","LP_Hal_PI_Br","LP_Hal_PI_Cl","LP_Hal_PI_I","LP_K_X","LP_Mg_X","LP_NH_F","LP_NH_Hal_Br","LP_NH_Hal_Cl","LP_NH_Hal_I","LP_NH_PI","LP_NH_S","LP_Na_X","LP_Ni_X","LP_OH_F","LP_OH_Hal_Br","LP_OH_Hal_Cl","LP_OH_Hal_I","LP_OH_PI","LP_OH_S","LP_OMulPol","LP_PI_PI","LP_Zn_X","LP_vdW"
    ]
    # fmt: on

    def __init__(
        self,
        param_path: str = os.path.join(int_desc_location, "sample/ligand/param.yaml"),
        vdw_radius_path: str = os.path.join(int_desc_location, "sample/ligand/vdw_radius.yaml"),
        priority_path: str = os.path.join(int_desc_location, "sample/ligand/priority.yaml"),
        water_definition_path: str = os.path.join(int_desc_location, "water_definition.txt"),
        interaction_group_path: str = os.path.join(int_desc_location, "group.yaml"),
        allow_mediate_pos: Optional[int] = None,
        on_14: bool = False,
        dup: bool = False,
        no_mediate: bool = False,
        no_out_total: bool = False,
        no_out_pml: bool = False,
        switch_ch_pi: bool = False,
    ):
        """
        Initialize the IntDescRunner with the given parameters.
        See intdesc documentation for more details.
        """
        self.intdesc_params = {
            "parametar_file": param_path,
            "vdw_file": vdw_radius_path,
            "priority_file": priority_path,
            "water_definition_file": water_definition_path,
            "interaction_group_file": interaction_group_path,
            "allow_mediate_position": allow_mediate_pos,
            "on_14": on_14,
            "dup": dup,
            "no_mediate": no_mediate,
            "no_out_total": no_out_total,
            "no_out_pml": no_out_pml,
            "switch_ch_pi": switch_ch_pi,
        }

    def compute_surrounding_resids(self, complex_struct: prody.AtomGroup, ligand_filter: str, distance: float):
        surroundings = complex_struct.select(
            f"protein and not (resname {ligand_filter}) and within {distance} of (resname {ligand_filter} and noh)"
        )
        # proDy start indices at 0 but intdesc expect indices to start at 1 so add 1 to avoid mismatch
        uniq_resid = np.unique(surroundings.getResnums()).tolist()
        if len(uniq_resid) == 0:
            raise ValueError(f"No residue found for the distance {distance} and the ligand provided: {ligand_filter}")
        return uniq_resid

    def compute_ligand_name(self, complex_struct: prody.AtomGroup, ligand_filter: str):
        return list(set(complex_struct.select(f"resname {ligand_filter}").getResnames().tolist()))

    def prepare_molecular_select(
        self, complex_struct: prody.AtomGroup, ligand_filter: str, distance: float, solvent_name: str, output_dir: str
    ) -> Path:

        param = {
            "ligand": {"name": self.compute_ligand_name(complex_struct, ligand_filter)},
            "protein": {"num": self.compute_surrounding_resids(complex_struct, ligand_filter, distance)},
            "solvent": {"name": solvent_name},
        }
        molcular_select_filepath = Path(output_dir) / "molecular_select.yaml"
        molcular_select_filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(str(molcular_select_filepath), "w") as file:
            yaml.dump(param, file, indent=4)

        return molcular_select_filepath

    def get_output_prefix(self, output_dir: str) -> str:
        """
        Get the output prefix for the intDesc calculation.
        Returns:
            str: The output prefix.
        """
        return str(Path(output_dir) / "output_intdesc")

    def run(
        self,
        mol2_filepath: str,
        complex_struct: prody.AtomGroup,
        ligand_filter: str,
        distance: float,
        solvent_name: str,
        output_dir: str,
    ) -> pd.DataFrame:
        """
        Run the intDesc calculation.

        Args:
            mol2_filepath (str): Path to the MOL2 file.
            complex_struct (prody.AtomGroup): The complex structure.
            ligand_filter (str): Ligand resname used as a filter in proDy.
            distance (float): The distance cutoff for residue to consider.
            solvent_name (str): The solvent name.
            output_dir (str): The output directory.

        Returns:
            pd.DataFrame: The interaction descriptor results.
        """
        molcular_select_filepath = self.prepare_molecular_select(
            complex_struct, ligand_filter, distance, solvent_name, output_dir
        )

        intdesc_output = StringIO()
        with redirect_stdout(intdesc_output):
            calculate(
                exec_type="Lig",
                mol2=str(mol2_filepath),
                molcular_select_file=str(molcular_select_filepath.absolute()),
                output=self.get_output_prefix(output_dir),
                **self.intdesc_params,
            )

        return self.extract_output(output_dir)

    def extract_output(self, output_dir: str) -> pd.DataFrame:
        """
        Extract the output from the intDesc calculation.

        Returns:
            pd.DataFrame: The interaction descriptor results.
        """
        output_prefix = self.get_output_prefix(output_dir)
        try:
            intdesc = pd.read_csv(output_prefix + "_one_hot_list.csv")
            intdesc[["atom_number", "partner_atom_number"]] = intdesc[["atom_number", "partner_atom_number"]] - 1
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"No interaction between Ligand and Protein found (intDesc result are empty)") from e
        return intdesc


class PDBtoMol2Converter:

    @staticmethod
    def get_mol2_filepath(outdir: str, pdb_filepath: str):
        return Path(outdir) / f"{Path(pdb_filepath).stem}.mol2"

    @staticmethod
    def pdb_to_mol2(pdb_filepath: str, outdir: str, overwrite: bool = True) -> str:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        mol2_filepath = PDBtoMol2Converter.get_mol2_filepath(outdir, pdb_filepath)

        if overwrite and mol2_filepath.exists():
            return mol2_filepath

        result = subprocess.run(["obabel", pdb_filepath, "-O", mol2_filepath], capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise ChildProcessError(f"Error while converting {pdb_filepath} to mol2: {error_msg}")

        return str(mol2_filepath)


class Mol2Processor:

    @staticmethod
    def extract_bond_indices_from_file(filepath):
        """
        Extracts bond information from a mol2 file as pairs of atom indices.

        Args:
            filepath (str): Path to the .mol2 file.

        Returns:
            list of tuple: A list of pairs (atom1_index, atom2_index) representing bonds.
        """
        bond_indices = []
        inside_bond_section = False

        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                # Check for the start of the bond section
                if line == "@<TRIPOS>BOND":
                    inside_bond_section = True
                    continue
                # Check for the end of the bond section
                elif line.startswith("@<TRIPOS>") and inside_bond_section:
                    break

                # Process lines inside the bond section
                if inside_bond_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        atom1 = int(parts[1]) - 1  # Convert 1-based index to 0-based
                        atom2 = int(parts[2]) - 1  # Convert 1-based index to 0-based
                        bond_indices.append((atom1, atom2))

        return bond_indices

    @staticmethod
    def extract_atom_types_from_file(filepath):
        """
        Extract atom types from a MOL2 file.

        Args:
            filepath (str): Path to the MOL2 file.

        Returns:
            list: A list of atom types in the MOL2 file.
        """
        atom_types = []
        in_atom_section = False

        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()

                # Check for the start of the ATOM section
                if line.startswith("@<TRIPOS>ATOM"):
                    in_atom_section = True
                    continue
                # Check for the end of the ATOM section
                elif line.startswith("@<TRIPOS>") and in_atom_section:
                    break

                # Extract atom type if in the ATOM section
                if in_atom_section and line:
                    columns = line.split()
                    atom_types.append(columns[5])

        return atom_types


class AntechamberRunner:

    # fmt: off
    # Taken from https://github.com/choderalab/ambermini/tree/master/share/amber/dat/antechamber + some missing one C.ar, C.3. H, cf
    additional_vocabulary = ['Al', 'Any', 'Br', 'C', 'C*', 'C.1', 'C.2', 'C.3', 'C.ar', 'C.cat', 'C1', 'CA', 'CB', 'CC', 'CD', 'CK', 'CM', 'CN', 'CQ', 'CR', 'CT', 'CV', 'CW', 'CY', 'CZ', 'Ca', 'Cl', 'DU', 'Du', 'F', 'H', 'H', 'H.spc', 'H.t3p', 'H1', 'H2', 'H3', 'H4', 'H5', 'HA', 'HC', 'HO', 'HP', 'HS', 'Hal', 'Het', 'Hev', 'I', 'K', 'LP', 'Li', 'N', 'N*', 'N.1', 'N.2', 'N.3', 'N.4', 'N.ar', 'N.pl3', 'N1', 'N2', 'N3', 'NA', 'NB', 'NC', 'NT', 'NY', 'Na', 'O', 'O.2', 'O.3', 'O.co2', 'O.spc', 'O.t3p', 'O2', 'OH', 'OS', 'OW', 'P', 'P.3', 'S', 'S.2', 'S.3', 'S.O', 'S.O2', 'SH', 'SO', 'Si', 'br', 'c', 'c1', 'c2', 'c3', 'ca', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cl', 'cp', 'cq', 'cu', 'cv', 'cx', 'cy', 'cz', 'du', 'f', 'h1', 'h2', 'h3', 'h4', 'h5', 'ha', 'hc', 'hn', 'ho', 'hp', 'hs', 'hw', 'hx', 'i', 'n', 'n', 'n1', 'n2', 'n3', 'n4', 'n7', 'n8', 'n9', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'nh', 'no', 'ns', 'nt', 'nu', 'nv', 'nx', 'ny', 'nz', 'o', 'oh', 'os', 'ow', 'p2', 'p3', 'p4', 'p5', 'pb', 'pc', 'pd', 'pe', 'pf', 'px', 'py', 's', 's2', 's4', 's6', 'sh', 'ss', 'sx', 'sy']
    gaff_vocabulary = ["Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Ca", "Cd", "Ce", "Cf", "Cm", "Co", "Cr", "Cs", "Cu", "C.ar", "C.3", "DU", "Db", "Ds", "Dy", "Er", "Es", "Eu", "Fe", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg", "Ho", "Hs", "In", "Ir", "K", "Kr", "LP", "La", "Li", "Lr", "Lu", "Md", "Mg", "Mn", "Mo", "Mt", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np", "Os", "O.3", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rh", "Rn", "Ru", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr", "br", "c", "c1", "c2", "c3", "ca", "cc", "ce", "cf", "cg", "ch", "cl", "cp", "cu", "cv", "cx", "cy", "cz", "f", "h1", "h2", "h3", "h4", "h5", "ha", "hc", "hn", "ho", "hp", "hs", "hw", "hx", "i", "lp", "n", "n1", "n2", "n3", "n4", "na", "nb", "nc", "ne", "nh", "no", "o", "oh", "os", "p2", "p3", "p4", "p5", "pb", "pc", "pe", "px", "py", "s", "s2", "s4", "s6", "sh", "ss", "sx", "sy"]
    bcc_vocabulary = {v: k for k,v in enumerate(["11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", "31", "32", "33", "42", "41", "51", "52", "53", "61", "71", "72", "73", "74", "91"])}
    pdb_vocabulary = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Th": 90, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}
    # Taken from http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    sybyl_vocabulary = {v: k for k,v in enumerate(["C.3","C.2","C.1","C.ar","C.cat","N.3","N.2","N.1","N.ar","N.am","N.pl3","N.4","Na","O.3","O.2","O.co2","O.spc","O.t3p","S.3","S.2","S.O","S.O2","P.3","F","H","H.spc","H.t3p","LP","Du","Du.C","Any","Hal","Het","Hev","Li","Na","Mg","Al","Si","K","Ca","Cr.th","Cr.oh","Mn","Fe","Co.oh","Cu","Cl","Br","I","Zn","Se","Mo","Sn"])}
    AT_VOCABULARY = {
        "AM1-BCC": bcc_vocabulary,
        "GAFF": {v: k for k,v in enumerate(np.unique(gaff_vocabulary + additional_vocabulary))},
        "PDB": pdb_vocabulary,
        "SYBYL": sybyl_vocabulary
    }
    # fmt: on

    def __init__(self):

        self.at = {"AM1-BCC": "bcc", "GAFF": "gaff", "SYBYL": "sybyl"}

    def get_antechamber_filepath(self, filepath_pdb: str, atom_type: str, outdir: str):
        return str(Path(outdir) / f"{Path(filepath_pdb).stem}_{self.at[atom_type]}.mol2")

    def run_antechamber(self, pdb_filepath: str, atom_type: str, outdir: str, overwrite: bool = True) -> str:
        """
        Run antechamber between the ligand pdb file and generate a mpdb file with the atom_type provided
        """
        Path(outdir).mkdir(parents=True, exist_ok=True)
        antechamber_filepath = self.get_antechamber_filepath(pdb_filepath, atom_type, outdir)
        if overwrite and Path(antechamber_filepath).exists():
            return antechamber_filepath

        cmd_str = f"antechamber -pf y -i {Path(pdb_filepath).absolute()} -fi pdb -o {Path(antechamber_filepath).name} -fo mol2 -at {self.at[atom_type]}"
        cmd = cmd_str.split(" ")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise ChildProcessError(f"Something went wrong with antechamber cmd: {cmd_str}, error: {error_msg}")
        return antechamber_filepath

    def get_antechamber_atom_types(
        self, filepath_pdb: str, atom_type: str, outdir: str, overwrite: bool = True
    ) -> List[str]:
        antechamber_path = self.run_antechamber(filepath_pdb, atom_type, outdir, overwrite)
        return Mol2Processor.extract_atom_types_from_file(antechamber_path)

    @staticmethod
    def tokenize_atom_types(atom_types: List[str], atom_type: str) -> List[int]:
        return np.array([AntechamberRunner.AT_VOCABULARY[atom_type][at] for at in atom_types])

    def get_multiple_tokenize_atom_types(
        self, filepath_pdb: str, atom_types: List[str], outdir: str, overwrite: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate tokenized atom types for multiple atom types from a PDB file.
        Args:
            filepath_pdb (str): Path to the PDB file.
            atom_types (List[str]): List of antechamber atom types to process.
            outdir (str): Output directory to store intermediate files.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.
        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are atom types and values are tokenized atom types.
        """
        antechamber_atom_types = {}
        for at in atom_types:
            at_atom_types = self.get_antechamber_atom_types(filepath_pdb, at, outdir, overwrite)

            tokenize_atom_types = AntechamberRunner.tokenize_atom_types(at_atom_types, at)
            antechamber_atom_types[at] = tokenize_atom_types

        return antechamber_atom_types
