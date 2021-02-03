import itertools
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, List, Tuple, Callable, Optional, Union
import logging

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data as TorchGeometricData

from lib.core.exceptions import FeaturizationError
from lib.data.resources import Data


class AbstractFeaturizer(metaclass=ABCMeta):

    def __init__(self, inputs: List[str], outputs: List[str]):
        self._inputs = inputs
        self._outputs = outputs

    @abstractmethod
    def _process(self, data: Any) -> Any:
        raise NotImplementedError

    def run(self, data: Data) -> Data:
        if len(self._inputs) != len(self._outputs):
            raise FeaturizationError("Inputs and mappings must have the same length.")

        for index in range(len(self._inputs)):
            raw_data = data.inputs.pop(self._inputs[index])
            data.inputs[self._outputs[index]] = self._process(raw_data)

        return data


class AbstractTorchGeometricFeaturizer(AbstractFeaturizer):
    """
    Featurizers preparing data for torch geometric should extend this class
    """

    def _process(self, data: str) -> TorchGeometricData:
        mol = Chem.MolFromSmiles(data)
        if mol is None:
            raise FeaturizationError("Could not featurize entry: [{}]".format(data))

        atom_features = self._get_vertex_features(mol)
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_attributes = torch.FloatTensor(edge_attributes).view(-1, len(edge_attributes[0]))

        if edge_indices.numel() > 0:  # Sort indices
            permutation = (edge_indices[0] * atom_features.size(0) + edge_indices[1]).argsort()
            edge_indices, edge_attributes = edge_indices[:, permutation], edge_attributes[permutation]

        return TorchGeometricData(
            x=atom_features, edge_index=edge_indices,
            edge_attr=edge_attributes, smiles=data, mol=mol
        )

    def _get_vertex_features(self, mol: Chem.Mol) -> List[List[float]]:
        return [self._featurize_atom(atom) for atom in mol.GetAtoms()]

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[List[List[int]], List[List[float]]]:
        edge_indices, edge_attributes = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices += [[i, j], [j, i]]
            bond_features = self._featurize_bond(bond)
            edge_attributes += [bond_features, bond_features]

        return edge_indices, edge_attributes

    @abstractmethod
    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        raise NotImplementedError


class LegacyGraphFeaturizer(AbstractTorchGeometricFeaturizer):
    """
    The base molecule featurizer found in pytorch geometric.
    The input is expected to be a SMILES string.

    Adapted from: torch_geometric.datasets.molecule_net.MoleculeNet::process()
    """
    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        from torch_geometric.datasets.molecule_net import x_map as atom_mappings

        return [
            atom_mappings['atomic_num'].index(atom.GetAtomicNum()),
            atom_mappings['chirality'].index(str(atom.GetChiralTag())),
            atom_mappings['degree'].index(atom.GetTotalDegree()),
            atom_mappings['formal_charge'].index(atom.GetFormalCharge()),
            atom_mappings['num_hs'].index(atom.GetTotalNumHs()),
            atom_mappings['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
            atom_mappings['hybridization'].index(str(atom.GetHybridization())),
            atom_mappings['is_aromatic'].index(atom.GetIsAromatic()),
            atom_mappings['is_in_ring'].index(atom.IsInRing())
        ]

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        from torch_geometric.datasets.molecule_net import e_map as bond_mappings

        return [
            bond_mappings['bond_type'].index(str(bond.GetBondType())),
            bond_mappings['stereo'].index(str(bond.GetStereo())),
            bond_mappings['is_conjugated'].index(bond.GetIsConjugated())
        ]


class GraphFeaturizer(AbstractTorchGeometricFeaturizer):
    """
    Improved featurizer for graph-based models
    """

    DEFAULT_ATOM_TYPES = ["B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Br", "I"]

    def __init__(self, inputs: List[str], outputs: List[str], allowed_atom_types: Optional[List[str]] = None):
        super().__init__(inputs, outputs)

        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES

        self._allowed_atom_types = allowed_atom_types

    def _process(self, data: str) -> TorchGeometricData:
        data = super()._process(data=data)

        molecule_features = self._featurize_molecule(data.mol)
        molecule_features = torch.FloatTensor(molecule_features).view(-1, len(molecule_features))

        data.molecule_features = molecule_features
        return data

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return list(itertools.chain.from_iterable(
            [featurizer(atom) for featurizer in self._list_atom_featurizers()]
        ))

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return list(itertools.chain.from_iterable(
            [featurizer(bond) for featurizer in self._list_bond_featurizers()]
        ))

    def _featurize_molecule(self, mol: Chem.Mol) -> List[float]:
        return [featurizer(mol) for featurizer in self._list_molecule_featurizers()]

    def _list_atom_featurizers(self) -> List[Callable]:
        # 45 features by default
        from vendor.dgllife.utils.featurizers import (
            atom_type_one_hot, atom_degree_one_hot, atom_implicit_valence_one_hot, atom_formal_charge,
            atom_num_radical_electrons, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot
        )

        return [
            partial(atom_type_one_hot, allowable_set=self._allowed_atom_types, encode_unknown=True),
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot
        ]

    def _list_bond_featurizers(self) -> List[Callable]:
        # 12 features
        from vendor.dgllife.utils.featurizers import (
            bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot
        )

        return [
            bond_type_one_hot,
            bond_is_conjugated,
            bond_is_in_ring,
            bond_stereo_one_hot
        ]

    def _list_molecule_featurizers(self) -> List[Callable]:
        # 10 features
        from rdkit.Chem import Descriptors, Lipinski, Crippen
        from rdkit.Chem.QED import qed

        return [
            Descriptors.MolWt,
            Descriptors.NumRadicalElectrons,
            Descriptors.NumValenceElectrons,
            Lipinski.HeavyAtomCount,
            Lipinski.NumHDonors,
            Lipinski.NumHAcceptors,
            Lipinski.NumAromaticRings,
            Lipinski.NumRotatableBonds,
            Crippen.MolLogP,
            qed
        ]


class AbstractFingerprintFeaturizer(AbstractFeaturizer):
    """Abstract featurizer for fingerprints"""

    @abstractmethod
    def _process(self, data: Any) -> Union[List[int], np.ndarray]:
        raise NotImplementedError


class CircularFingerprintFeaturizer(AbstractFingerprintFeaturizer):
    """Morgan fingerprint featurizer"""

    def __init__(self, inputs: List[str], outputs: List[str], fingerprint_size: int = 2048, radius: int = 2):
        super().__init__(inputs, outputs)

        self._fingerprint_size = fingerprint_size
        self._radius = radius

    def _process(self, data: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(data)
        if mol is None:
            raise FeaturizationError("Could not featurize entry: [{}]".format(data))

        return self._generate_fingerprint(mol)

    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        from rdkit.Chem import AllChem

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self._radius, nBits=self._fingerprint_size)

        features = np.zeros(self._fingerprint_size, dtype=np.uint8)
        features[fingerprint.GetOnBits()] = 1

        return features


class OneHotEncoderFeaturizer(AbstractFeaturizer):
    """One-Hot encode a single string"""

    def __init__(self, inputs: List[str], outputs: List[str], classes: List[str]):
        super().__init__(inputs, outputs)

        self._classes = classes

    def _process(self, data: str) -> np.ndarray:
        features = np.zeros(len(self._classes))
        features[self._classes.index(data)] = 1

        return features


class TokenFeaturizer(AbstractFeaturizer):
    """Similar to the one-hot encoder, but will tokenize a whole sentence."""

    def __init__(self, inputs: List[str], outputs: List[str], classes: List[str], length: int, separator: str = ""):
        super().__init__(inputs, outputs)

        self._classes = classes
        self._separator = separator
        self._length = length

    def _process(self, data: str) -> np.ndarray:
        tokens = data.split(self._separator)
        features = np.zeros((self._length, len(self._classes)))

        for index in range(len(tokens)):
            if index == self._length:
                logging.warning("[CAUTION] Input is out of bounds. Features will be trimmed. --- {}".format(data))
                break

            features[index][self._classes.index(tokens[index])] = 1

        return features
