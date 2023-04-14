from abc import ABCMeta, abstractmethod
import math
import random
from copy import deepcopy
from typing import Dict, List, Optional
import itertools

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from scipy.stats import bernoulli
import torch
from torch_geometric.data import Data as PyG_Data
import numpy as np

from ..vendor.openfold.utils.tensor_utils import tensor_tree_map

ATOM_LIST = list(range(1,120)) # Includes mask token
NUM_ATOM_TYPE = len(ATOM_LIST)

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
NUM_CHIRALITY_TAG = len(CHIRALITY_LIST)

BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC,
    BT.UNSPECIFIED,
]
NUM_BOND_TYPE = len(BOND_LIST) + 1 # including aromatic and self-loop edge


BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
NUM_BOND_DIRECTION = len(BONDDIR_LIST)


class AbstractAugmentation(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data: Dict, seed=None) -> None:
        raise NotImplementedError

class BaseTransform(AbstractAugmentation):
    def __init__(self, prob: float = 1.0, input_field=None):
        """
        @param p: the probability of the transform being applied; default value is 1.0. If
                  a list is passed in, the value will be randomly and sampled between the two
                  end points.
        @param input_field: the name of the generated input containing the smile information
            should be a PyG_Data object
        """
        if(isinstance(prob, list)):
            assert 0 <= prob[0] <= 1.0
            assert 0 <= prob[1] <= 1.0
            assert prob[0] < prob[1]
        else:
            assert 0 <= prob <= 1.0, "p must be a value in the range [0, 1]"
        self.prob = prob
        self.input_field = input_field


    def __call__(self, data: Dict, seed=None) -> PyG_Data:
        """
        @param mol_graph: PyG Data to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PyG Data
        """
        if(isinstance(self.prob, list)):
            self.p = random.uniform(self.prob[0], self.prob[1])
        else:
            self.p = self.prob

        if self.input_field is not None:
            original = deepcopy(data)
            mol_graph = data[self.input_field]

        assert isinstance(self.p, (float, int))
        assert isinstance(mol_graph, PyG_Data), "mol_graph passed in must be a PyG Data"
        output = self.apply_transform(mol_graph, seed)
        if self.input_field is not None:
            original[self.input_field] = output
            output = original
        return output

    def apply_transform(self, mol_graph: PyG_Data) -> PyG_Data:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()


class RandomAtomMaskAugmentation(BaseTransform):
    """
    Base logic taken form Auglichem https://baratilab.github.io/AugLiChem/molecule.html
    """
    def __init__(self, p: float = 1.0, input_field=None):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        @param input_field: the name of the generated input containing the smile information
            should be a PyG_Data object
        """
        super().__init__(p, input_field)

    def apply_transform(self, mol_graph: PyG_Data, seed: Optional[None] = None) -> PyG_Data:
        """
        Transform that randomly mask atoms given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @param seed:
        @returns: Augmented PyG Data
        """
        if(seed is not None):
            random.seed(seed)
        N = mol_graph.x.size(0)
        num_mask_nodes = max([1, math.floor(self.p*N)])
        mask_nodes = random.sample(list(range(N)), num_mask_nodes)

        aug_mol_graph = deepcopy(mol_graph)
        for atom_idx in mask_nodes:
            aug_mol_graph.x[atom_idx,:] = torch.tensor(aug_mol_graph.x[atom_idx,:].shape)

        return aug_mol_graph

    def __str__(self):
        return "RandomAtomMask(p = {})".format(self.prob)


class RandomBondDeleteAugmentation(BaseTransform):
    """
    Base logic taken form Auglichem https://baratilab.github.io/AugLiChem/molecule.html
    """
    def __init__(self, p: float = 1.0, input_field=None):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        @param input_field: the name of the generated input containing the smile information
            should be a PyG_Data object
        """
        super().__init__(p, input_field)

    def apply_transform(self, mol_graph: PyG_Data, seed: Optional[None] = None) -> PyG_Data:
        """
        Transform that randomly delete chemical bonds given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @returns: Augmented PyG Data
        """
        if len(mol_graph.edge_attr.shape) == 1: # No edge case
            return mol_graph
        if(seed is not None):
            random.seed(seed)
        M = mol_graph.edge_index.size(1) // 2
        num_mask_edges = max([0, math.floor(self.p*M)])
        mask_edges_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges = [2*i for i in mask_edges_single] + [2*i+1 for i in mask_edges_single]

        aug_mol_graph = deepcopy(mol_graph)
        aug_mol_graph.edge_index = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.int64)
        aug_mol_graph.edge_attr = torch.zeros((2*(M-num_mask_edges), aug_mol_graph.edge_attr.size(1)), dtype=torch.float32)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges:
                aug_mol_graph.edge_index[:,count] = mol_graph.edge_index[:,bond_idx]
                aug_mol_graph.edge_attr[count,:] = mol_graph.edge_attr[bond_idx,:]
                count += 1

        return aug_mol_graph

    def __str__(self):
        return "RandomBondDelete(p = {})".format(self.prob)


class RandomTemplateSelectionAugmentation(AbstractAugmentation):

    def __call__(self, data: Dict, seed=None)-> None:
        if type(data['protein']) == dict:
            temple_number = np.random.random_integers(0, data['protein']["aatype"].shape[-1] - 1)
            fetch_cur_batch = lambda t: t[..., temple_number]
            data['protein'] = tensor_tree_map(fetch_cur_batch, data['protein'])
        elif type(data['protein']) == list:
            temple_number = np.random.random_integers(0, len(data['protein']) -1)
            data['protein'] = data['protein'][temple_number][0]
        return data


class ProteinPertubationAugmentation(AbstractAugmentation):
    def __init__(self, vocabulary: List[str], p: float = 0.2):
        self._original_vocabulary = vocabulary
        self._bernoulli_prob = p

    def _perturb(self, data: str) -> str:
        bern = bernoulli(self._bernoulli_prob)
        bern_seq = bern.rvs(len(data))
        l_data = list(data)
        for i in range(len(data)):
            if bern_seq[i]:
                l_data[i] = random.choice(self._original_vocabulary)
        return ''.join(l_data)

class ProteinPerturbationBaggedAugmentation(ProteinPertubationAugmentation):
    """
    Augmentation useful for pseudo lrodd background network. Perturb the protein following a Bernoulli distribution.
    """
    def __init__(self, vocabulary: List[str], max_length: int, p: float = 0.2, input: str = "target_sequence", output: str = "protein"):
        self._original_vocabulary = vocabulary
        self._bernoulli_prob = p
        self._vocabulary = self._get_combinations(vocabulary, max_length)
        self._max_length = max_length
        self._input = input
        self._output = output

    def _get_combinations(self, vocabulary: List[str], max_length: int) -> List[str]:
        combinations = []

        for length in range(1, max_length + 1):
            for variation in itertools.product(vocabulary, repeat=length):
                combinations.append("".join(variation))

        return combinations

    def __call__(self, data: dict, seed=None) -> dict:
        target_sequence = data[self._input]
        target_sequence = self._perturb(target_sequence)

        sample = dict.fromkeys(self._vocabulary, 0)

        for length in range(1, self._max_length + 1):
            for start_index in range(0, len(target_sequence) - length + 1):
                sample[target_sequence[start_index:start_index + length]] += 1

        data[self._output] = torch.FloatTensor(list(sample.values()))
        return data

class ProteinPerturbationSequenceAugmentation(ProteinPertubationAugmentation):
    """
    Augmentation useful for lrodd generative bg network. Perturb the protein following a Bernoulli distribution.
    """
    def __init__(self, vocabulary: List[str], p: float = 0.2, input: str = "target_sequence", output: str = "protein_index"):
        self._original_vocabulary = vocabulary
        self._bernoulli_prob = p
        self._to_index_dict = self._create_index_dict()
        self._input = input
        self._output = output

    def _create_index_dict(self):
        to_index_dict = {}
        for index, amino_acid in enumerate(self._original_vocabulary):
            to_index_dict[amino_acid] = index

        return to_index_dict

    def __call__(self, data: dict, seed=None) -> dict:
        target_sequence = data[self._input]
        target_sequence = self._perturb(target_sequence)
        data[self._output] = [self._to_index_dict[amino_acid] for amino_acid in target_sequence]
        return data