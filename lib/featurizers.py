from abc import abstractmethod, ABCMeta
from typing import Any

import numpy as np
from rdkit import Chem


class AbstractFeaturizer(metaclass=ABCMeta):

    def __init__(self, max_atoms: int):
        self.max_atoms = max_atoms

    @abstractmethod
    def apply(self, mol: Chem.Mol) -> Any:
        raise NotImplementedError


class AtomFeaturizer(AbstractFeaturizer):

    CHIRAL_TAGS = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']
    HYBRIDIZATIONS = ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']

    def apply(self, mol: Chem.Mol) -> np.ndarray:

        if mol.GetNumAtoms() > self.max_atoms:
            raise ValueError("Atom count exceeds maximum allowed value: {}".format(self.max_atoms))

        features = []
        for atom in mol.GetAtoms():
            features.append(
                [
                    atom.GetAtomicNum(),
                    self.CHIRAL_TAGS.index(atom.GetChiralTag()),
                    atom.GetTotalDegree(),
                    atom.GetFormalCharge(),
                    atom.GetTotalNumHs(),
                    atom.GetNumRadicalElectrons(),
                    self.HYBRIDIZATIONS.index(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    int(atom.IsInRing())
                ]
            )

        features = np.array(features)
        features = np.pad(features, pad_width=((0, self.max_atoms - features.shape[0]), (0, 0)))

        return features

    def get_feature_count(self) -> int:
        return 9


class AdjacencyMatrixFeaturizer(AbstractFeaturizer):

    def apply(self, mol: Chem.Mol) -> np.ndarray:

        if mol.GetNumAtoms() > self.max_atoms:
            raise ValueError("Atom count exceeds maximum allowed value: {}".format(self.max_atoms))

        adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])

        degree_matrix = adjacency_matrix.sum(axis=1)
        degree_matrix = np.power(degree_matrix, -0.5).flatten()
        degree_matrix = np.diag(degree_matrix)

        adjacency_matrix = np.matmul(adjacency_matrix, degree_matrix)
        adjacency_matrix = adjacency_matrix.transpose()
        adjacency_matrix = np.matmul(adjacency_matrix, degree_matrix)

        padding_size = self.max_atoms - adjacency_matrix.shape[0]
        adjacency_matrix = np.pad(adjacency_matrix, pad_width=((0, padding_size), (0, padding_size)))

        return adjacency_matrix
