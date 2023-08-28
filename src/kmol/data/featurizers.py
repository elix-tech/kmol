import os
import io
import itertools
from pathlib import Path
import pyximport
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pickle
import yaml
import subprocess


import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.data import Data as TorchGeometricData
from scipy.stats import bernoulli

from kmol.data.resources import DataPoint

from .resources import DataPoint
from ..core.exceptions import FeaturizationError
from ..core.helpers import SuperFactory
from ..core.logger import LOGGER as logging
from ..vendor.openfold.data import templates, data_pipeline, feature_pipeline
from ..vendor.openfold.config import model_config
from ..vendor.openfold.utils.tensor_utils import tensor_tree_map
from ..model.architectures import AlphaFold
from .loaders import ListLoader

import algos


class AbstractFeaturizer(metaclass=ABCMeta):
    def __init__(self, inputs: List[str], outputs: List[str], should_cache: bool = False, rewrite: bool = True):
        self._inputs = inputs
        self._outputs = outputs

        self._should_cache = should_cache
        self._rewrite = rewrite

        self.__cache = {}

    @abstractmethod
    def _process(self, data: Any, entry: DataPoint) -> Any:
        raise NotImplementedError

    def __process(self, data: Any, entry: DataPoint) -> Any:
        if self._should_cache:
            if data not in self.__cache:
                self.__cache[data] = self._process(data, entry)

            return self.__cache[data]
        else:
            return self._process(data, entry)

    def set_device(self, device: torch.device):
        self.device = device

    def run(self, data: DataPoint) -> None:
        try:
            if len(self._inputs) != len(self._outputs):
                raise FeaturizationError("Inputs and mappings must have the same length.")

            for index in range(len(self._inputs)):
                raw_data = data.inputs[self._inputs[index]]
                if self._rewrite:
                    data.inputs.pop(self._inputs[index])

                data.inputs[self._outputs[index]] = self.__process(raw_data, data)
        except (
            FeaturizationError,
            ValueError,
            IndexError,
            AttributeError,
            TypeError,
        ) as e:
            raise FeaturizationError(
                "[WARNING] Could not run featurizer '{}' on '{}' --- {}".format(self.__class__.__name__, data.id_, e)
            )


class AbstractTorchGeometricFeaturizer(AbstractFeaturizer):
    """
    Featurizers preparing data for torch geometric should extend this class
    """

    def _process(self, data: str, entry: DataPoint) -> TorchGeometricData:
        mol = Chem.MolFromSmiles(data)
        if mol is None:
            raise FeaturizationError("Could not featurize entry: [{}]".format(data))

        atom_features = self._get_vertex_features(mol)
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_attributes = torch.FloatTensor(edge_attributes)

        if edge_indices.numel() > 0:  # Sort indices
            permutation = (edge_indices[0] * atom_features.size(0) + edge_indices[1]).argsort()
            edge_indices, edge_attributes = edge_indices[:, permutation], edge_attributes[permutation]

        return TorchGeometricData(x=atom_features, edge_index=edge_indices, edge_attr=edge_attributes, smiles=data)

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


class AbstractDescriptorComputer(metaclass=ABCMeta):
    @abstractmethod
    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[float]:
        raise NotImplementedError


class RdkitDescriptorComputer(AbstractDescriptorComputer):
    def _get_descriptor_calculators(self) -> List[Callable]:
        from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf, GraphDescriptors, rdMolDescriptors, QED

        return [
            Descriptors.MolWt,
            Descriptors.NumRadicalElectrons,
            Descriptors.NumValenceElectrons,
            rdMolDescriptors.CalcTPSA,
            MolSurf.LabuteASA,
            GraphDescriptors.BalabanJ,
            Lipinski.RingCount,
            Lipinski.NumAliphaticRings,
            Lipinski.NumSaturatedRings,
            Lipinski.NumRotatableBonds,
            Lipinski.NumHeteroatoms,
            Lipinski.HeavyAtomCount,
            Lipinski.NumHDonors,
            Lipinski.NumHAcceptors,
            Lipinski.NumAromaticRings,
            Crippen.MolLogP,
            QED.qed,
        ]

    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[Union[int, float]]:
        return [featurizer(mol) for featurizer in self._get_descriptor_calculators()]


class MordredDescriptorComputer(AbstractDescriptorComputer):
    def __init__(self):
        from mordred import Calculator, descriptors

        self._calculator = Calculator(descriptors, ignore_3D=True)

    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[Union[int, float]]:
        descriptors = self._calculator(mol)

        return list(descriptors.fill_missing(0))


class DescriptorFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        descriptor_calculator: AbstractDescriptorComputer,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)
        self._descriptor_calculator = descriptor_calculator

    def _process(self, data: str):
        mol = Chem.MolFromSmiles(data)
        molecule_features = self._descriptor_calculator.run(mol)
        return torch.FloatTensor(molecule_features)


class DatasetDescriptorComputer(AbstractDescriptorComputer):
    def __init__(self, targets: List[str]):
        self.targets = targets

    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[Union[int, float]]:
        return [entry.inputs[target] for target in self.targets]


class CombinedDescriptorComputer(AbstractDescriptorComputer):
    def __init__(self, calculators: List[Dict[str, Any]]):
        self.calculators = [SuperFactory.create(AbstractDescriptorComputer, options) for options in calculators]

    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[Union[int, float]]:
        return list(chain.from_iterable(calculator.run(mol, entry) for calculator in self.calculators))


class GraphFeaturizer(AbstractTorchGeometricFeaturizer):
    """
    Improved featurizer for graph-based models
    """

    DEFAULT_ATOM_TYPES = ["B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Br", "I"]

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        descriptor_calculator: AbstractDescriptorComputer,
        allowed_atom_types: Optional[List[str]] = None,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES

        self._allowed_atom_types = allowed_atom_types
        self._descriptor_calculator = descriptor_calculator

    def _process(self, data: str, entry: DataPoint) -> TorchGeometricData:
        mol = Chem.MolFromSmiles(data)
        data = super()._process(data=data, entry=entry)

        molecule_features = self._descriptor_calculator.run(mol, entry)
        molecule_features = torch.FloatTensor(molecule_features).view(-1, len(molecule_features))

        data.molecule_features = molecule_features
        return data

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return list(itertools.chain.from_iterable([featurizer(atom) for featurizer in self._list_atom_featurizers()]))

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return list(itertools.chain.from_iterable([featurizer(bond) for featurizer in self._list_bond_featurizers()]))

    def _list_atom_featurizers(self) -> List[Callable]:
        # 45 features by default
        from ..vendor.dgllife.utils.featurizers import (
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
        )

        return [
            partial(atom_type_one_hot, allowable_set=self._allowed_atom_types, encode_unknown=True),
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
        ]

    def _list_bond_featurizers(self) -> List[Callable]:
        # 12 features
        from ..vendor.dgllife.utils.featurizers import (
            bond_type_one_hot,
            bond_is_conjugated,
            bond_is_in_ring,
            bond_stereo_one_hot,
        )

        return [bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot]


class ChiralGraphFeaturizer(GraphFeaturizer):
    def _list_atom_featurizers(self) -> List[Callable]:
        from ..vendor.dgllife.utils.featurizers import (
            atom_chiral_tag_one_hot,
            atom_chirality_type_one_hot,
            atom_is_chiral_center,
        )

        featurizers = super()._list_atom_featurizers()

        featurizers.extend([atom_chiral_tag_one_hot, atom_chirality_type_one_hot, atom_is_chiral_center])

        return featurizers


class AbstractFingerprintFeaturizer(AbstractFeaturizer):
    """Abstract featurizer for fingerprints"""

    @abstractmethod
    def _process(self, data: Any, entry: DataPoint) -> Union[List[int], np.ndarray]:
        raise NotImplementedError


class CircularFingerprintFeaturizer(AbstractFingerprintFeaturizer):
    """Morgan fingerprint featurizer"""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        should_cache: bool = False,
        rewrite: bool = True,
        fingerprint_size: int = 2048,
        radius: int = 2,
        use_chirality: bool = False,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._fingerprint_size = fingerprint_size
        self._radius = radius
        self._use_chirality = use_chirality

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        mol = Chem.MolFromSmiles(data)
        if mol is None:
            raise FeaturizationError("Could not featurize entry: [{}]".format(data))

        return torch.FloatTensor(self._generate_fingerprint(mol))

    @staticmethod
    def generate_fingerprint(mol: Chem.Mol, radius: int, fingerprint_size: int, use_chirality: bool) -> np.ndarray:
        from rdkit.Chem import AllChem

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=fingerprint_size, useChirality=use_chirality
        )

        features = np.zeros(fingerprint_size, dtype=np.uint8)
        features[fingerprint.GetOnBits()] = 1

        return features

    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        return CircularFingerprintFeaturizer.generate_fingerprint(
            mol, self._radius, self._fingerprint_size, self._use_chirality
        )


class CircularFingerprintDescriptorComputer(AbstractDescriptorComputer):
    """Descriptor version of CircularFingperintFeaturizer"""

    def __init__(self, fingerprint_size: int = 2048, radius: int = 2, use_chirality: bool = False):
        self._fingerprint_size = fingerprint_size
        self._radius = radius
        self._use_chirality = use_chirality

    def run(self, mol: Chem.Mol, entry: DataPoint) -> List[Union[int, float]]:
        return list(self._generate_fingerprint(mol))

    def _generate_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        return CircularFingerprintFeaturizer.generate_fingerprint(
            mol, self._radius, self._fingerprint_size, self._use_chirality
        )


class OneHotEncoderFeaturizer(AbstractFeaturizer):
    """One-Hot encode a single string"""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        classes: List[str],
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._classes = classes

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        features = np.zeros(len(self._classes))
        features[self._classes.index(data)] = 1

        return torch.FloatTensor(features)


class TokenFeaturizer(AbstractFeaturizer):
    """Similar to the one-hot encoder, but will tokenize a whole sentence."""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        vocabulary: List[str],
        max_length: int,
        separator: str = "",
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._vocabulary = vocabulary
        self._separator = separator
        self._max_length = max_length

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        tokens = data.split(self._separator) if self._separator else [character for character in data]
        features = np.zeros((self._max_length, len(self._vocabulary)))

        for index, token in enumerate(tokens):
            if index == self._max_length:
                logging.warning("[CAUTION] Input is out of bounds. Features will be trimmed. --- {}".format(data))
                break

            features[index][self._vocabulary.index(token)] = 1

        return torch.FloatTensor(features)


class BagOfWordsFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        vocabulary: List[str],
        max_length: int,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)
        self._vocabulary = self._get_combinations(vocabulary, max_length)
        self._max_length = max_length

    def _get_combinations(self, vocabulary: List[str], max_length: int) -> List[str]:
        combinations = []

        for length in range(1, max_length + 1):
            for variation in itertools.product(vocabulary, repeat=length):
                combinations.append("".join(variation))

        return combinations

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        sample = dict.fromkeys(self._vocabulary, 0)

        for length in range(1, self._max_length + 1):
            for start_index in range(0, len(data) - length + 1):
                sample[data[start_index : start_index + length]] += 1

        return torch.FloatTensor(list(sample.values()))


class PerturbedBagOfWordsFeaturizer(BagOfWordsFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        vocabulary: List[str],
        max_length: int,
        should_cache: bool = False,
        rewrite: bool = True,
        perturbation: bool = False,
        bernoulli_prob: float = 0.2,
    ):
        super().__init__(inputs, outputs, vocabulary, max_length, should_cache, rewrite)
        self._original_vocabulary = vocabulary
        self._perturbation = perturbation
        self._bernoulli_prob = bernoulli_prob

    def _perturb(self, data: str) -> str:
        bern = bernoulli(self._bernoulli_prob)
        bern_seq = bern.rvs(len(data))
        l_data = list(data)
        for i in range(len(data)):
            if bern_seq[i]:
                l_data[i] = random.choice(self._original_vocabulary)
        return "".join(l_data)

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        if self._perturbation:
            data = self._perturb(data)

        sample = dict.fromkeys(self._vocabulary, 0)

        for length in range(1, self._max_length + 1):
            for start_index in range(0, len(data) - length + 1):
                sample[data[start_index : start_index + length]] += 1

        return torch.FloatTensor(list(sample.values()))


class FASTAFeaturizer(BagOfWordsFeaturizer):
    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        data = data.split("\n")[-1]
        return super()._process(data, entry)


class TransposeFeaturizer(AbstractFeaturizer):
    def _process(self, data: torch.Tensor, entry: DataPoint) -> torch.Tensor:
        return data.transpose(-1, -2)


class FixedFeaturizer(AbstractFeaturizer):
    def __init__(
        self, inputs: List[str], outputs: List[str], value: float, should_cache: bool = False, rewrite: bool = True
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)
        self._value = value

    def _process(self, data: float, entry: DataPoint) -> float:
        return round(data / self._value, 8)


class ConverterFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        source_format: str,
        target_format: str,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._source_format = source_format
        self._target_format = target_format

        from openbabel import pybel

        self._pybel = pybel

    def _process(self, data: str, entry: DataPoint) -> str:
        return self._pybel.readstring(self._source_format, data).write(self._target_format).strip()


class MsaFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        sequence_column: str,
        template_mmcif_dir: str,
        precompute_alignment_dir: str,
        crop_size: int,
        name_config: str = None,
        should_cache: bool = False,
        rewrite: bool = True,
        msa_extrator_cfg: Dict = None,
    ):
        """
        @param template_mmcif_dir: path to alphafold templates usually in alphafold_dataset/pdb_mmcif/mmcif_files/
        @param precompute_alignment_dir: path to the output os the GenerateMSA script dir, folder containing the alignment for all the proteins.
        @param crop_size: Size of the protein sequence after preprocessing.
        @param name_config: Name of the openfold configuration to load
            One can view the various possible name in
            `src/kmol/vendor/openfold/config.py` in the method `model_config`
        @param msa_extrator_cfg: Config for Alphafold model.
        """

        super().__init__(inputs, outputs, should_cache, rewrite)

        self.msa_extrator_cfg = msa_extrator_cfg
        if msa_extrator_cfg is not None:
            torch.multiprocessing.set_start_method("spawn")
            assert not msa_extrator_cfg.get("finetune", False), "In featurizer mode the extractor can't be finetune"
            self.msa_extrator = AlphaFold(**msa_extrator_cfg)
            self.msa_extrator.eval()
            # self.msa_extrator.to("cuda")

        if msa_extrator_cfg is not None:
            self.config = self.msa_extrator.cfg
        else:
            self.config = model_config(name_config)

        self.config.data.predict.crop_size = crop_size

        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date="2022-11-03",
            max_hits=self.config.data.predict.max_template_hits,
            kalign_binary_path="",
        )

        self.data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        self.sequence_column = sequence_column
        self.alignment_dir = precompute_alignment_dir

    def generate_feature_dict(
        self,
        tags,
        seqs,
    ):
        tmp_fasta_path = os.path.join("/tmp/", f"tmp_{os.getpid()}.fasta")
        if len(seqs) == 1:
            tag = tags[0]
            seq = seqs[0]
            with open(tmp_fasta_path, "w") as fp:
                fp.write(f">{tag}\n{seq}")

            local_alignment_dir = os.path.join(self.alignment_dir, tag)
            feature_dict = self.data_processor.process_fasta(fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir)
        else:
            with open(tmp_fasta_path, "w") as fp:
                fp.write("\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
            feature_dict = self.data_processor.process_multiseq_fasta(
                fasta_path=tmp_fasta_path,
                super_alignment_dir=self.alignment_dir,
            )

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)

        return feature_dict

    def _process(self, data, entry: DataPoint):
        sequence = entry.inputs.pop(self.sequence_column)
        feature_dict = self.generate_feature_dict(tags=[data], seqs=[sequence])
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict,
            mode="predict",
        )

        if self.msa_extrator_cfg is not None:
            self.msa_extrator.to(self.device)
            features = []
            with torch.no_grad():
                processed_feature_dict
                features = self.msa_extrator(tensor_tree_map(lambda x: x.to(self.device), processed_feature_dict))
                return features
        return processed_feature_dict

    def compute_unique_dataset_to_cache(self, loader: ListLoader):
        """
        Return a unique set of protein to cache in the featurizer.
        """
        data_source = loader._dataset
        try:
            data_source.pop("index")
        except KeyError:
            pass
        data_source = loader._dataset.reset_index()
        unique_indices = list(data_source.groupby(self.sequence_column).apply(lambda x: x.iloc[0]).loc[:, "index"].values)
        return Subset(loader, unique_indices)


class GraphormerFeaturizer(GraphFeaturizer):
    """
    Converts graph features into a format supported by Graphormer:
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        allowed_atom_types: Optional[List[str]] = None,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super(GraphFeaturizer, self).__init__(inputs, outputs, should_cache, rewrite)

        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES

        self._allowed_atom_types = allowed_atom_types

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        data = super(GraphFeaturizer, self)._process(data=data, entry=entry)
        return self._preprocess_item(data)

    def _convert_to_single_emb(self, x, offset: int = 512):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
        x = x + feature_offset
        return x

    def _list_atom_featurizers(self) -> List[Callable]:
        # 45 features by default
        from ..vendor.dgllife.utils.featurizers import atom_type_one_hot

        return [partial(atom_type_one_hot, allowable_set=self._allowed_atom_types, encode_unknown=True)]

    def _list_bond_featurizers(self) -> List[Callable]:
        # 12 features
        from ..vendor.dgllife.utils.featurizers import bond_type_one_hot

        return [bond_type_one_hot]

    def _preprocess_item(self, item):
        edge_attr, edge_index, x = item.edge_attr.long(), item.edge_index, item.x
        N = item.num_nodes
        if len(edge_attr) == 0:
            edge_attr = torch.zeros([N, 1]).long()
            edge_index = torch.zeros([2, N]).long()

        # if N <= 1:
        #     raise FeaturizationError("Molecule : [{}]".format(item))
        x = torch.argmax(item.x, 1)
        x = self._convert_to_single_emb(x.reshape(1, -1).t())

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        edge_attr = torch.argmax(edge_attr, 1)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.shape[1]], dtype=torch.long)
        attn_edge_type[edge_index[0].long(), edge_index[1].long()] = self._convert_to_single_emb(edge_attr) + 1

        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

        # combine
        item.x = x.long()
        item.adj = adj
        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree  # for undirected graph
        item.edge_input = torch.from_numpy(edge_input).long()

        return item

    def reverse(self, data: DataPoint) -> None:
        pass


class LoadFeaturizer(AbstractFeaturizer):
    """
    This class is used to load a set of feature computed outside kmol.
    This is an abstract class.
    """

    def __init__(self, inputs: List[str], outputs: List[str], folder_path: str, rewrite: bool = True, suffix: str = None):
        super().__init__(inputs, outputs, should_cache=False, rewrite=rewrite)
        self.folder_path = Path(folder_path)
        self.suffix = suffix

    def _process(self, data: str, entry: DataPoint) -> Any:
        path = self.folder_path / data
        if self.suffix is not None:
            path = path.parent / (path.name + self.suffix)
        return self.load(path)

    @abstractmethod
    def load(self, path) -> Any:
        raise NotImplementedError()


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class PickleLoadFeaturizer(LoadFeaturizer):
    def load(self, path):
        with open(path, "rb") as file:
            return CPU_Unpickler(file).load()


class NumpyLoadFeaturizer(LoadFeaturizer):
    def __init__(self, skip_npz_file: List[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_npz_file = skip_npz_file if skip_npz_file is not None else []

    def load(self, path):
        if path.suffix == ".npz":
            np_archive = np.load(path)
            return {file: np_archive[file] for file in np_archive.files if file not in self.skip_npz_file}
        if path.suffix == ".npy":
            return np.load(path)


class PdbqtToPdbFeaturizer(AbstractFeaturizer):
    """Change a pdbqt file to pdb"""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        pdqt_dir: str,
        dir_to_save: str = "/tmp/kmol/pdb",
        protonize: bool = False,
        ph: float = 7.0,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        from openbabel import openbabel as ob

        self._ob = ob
        self.outdir = Path(dir_to_save)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        self.pdqt_dir = pdqt_dir
        self.protonize = protonize
        self.ph = ph

    def get_pdqt_filepath(self, entry: DataPoint) -> Path:
        pdbqt_filepath = Path(entry["path"]) if "path" in entry.inputs else None

        if pdbqt_filepath is None:
            if "target_accession" in entry.inputs and "drug_id" in entry.inputs:
                filename = f"{entry.inputs['target_accession']}-drug_id-{entry.inputs['drug_id']}_best_model"
                pdbqt_filepath = Path(f"{self.pdqt_dir}/{filename}.pdbqt")
            else:
                raise FeaturizationError("No target_accession or drug_id found in data")

        if not pdbqt_filepath.exists():
            raise FeaturizationError(f"File {pdbqt_filepath} does not exist")

        return pdbqt_filepath

    def _process(self, data: str, entry: DataPoint) -> str:
        pdbqt_filepath = self.get_pdqt_filepath(entry)
        pdb_filepath = Path(f"{self.outdir}/{pdbqt_filepath.stem}.pdb")

        # Convert PDBQT to PDB
        conv = self._ob.OBConversion()
        conv.SetInAndOutFormats("pdbqt", "pdb")
        ob_mol = self._ob.OBMol()
        conv.ReadFile(ob_mol, str(pdbqt_filepath))
        if self.protonize:
            ob_mol.CorrectForPH(self.ph)
            ob_mol.AddHydrogens()
        if conv.WriteFile(ob_mol, str(pdb_filepath)):
            return str(pdb_filepath)
        else:
            raise FeaturizationError(f"Error while writing the pdb file for {str(pdbqt_filepath)}")


class PdbToMol2Featurizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        protein_atom_type: List[str] = ["SYBYL"],
        ligand_atom_type: List[str] = ["SYBYL"],
        dir_to_save: str = "/tmp/kmol/mol2",
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        from openbabel import openbabel as ob
        from moleculekit.molecule import Molecule

        self._ob = ob
        self._molecule = Molecule

        self.outdir = Path(dir_to_save)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        self.protein_atom_type = [s.upper() for s in protein_atom_type]
        self.ligand_atom_type = [s.upper() for s in ligand_atom_type]
        self.at = {"AM1-BCC": "bcc", "GAFF": "gaff"}
        self.need_antechamber = any([at in self.at for at in protein_atom_type])

    def _process(self, data: Any, entry: DataPoint) -> str:
        pdb_filepath = Path(data)
        mol2_filepath = Path(f"{self.outdir}/{pdb_filepath.stem}.mol2")

        ob_mol = self._ob.OBMol()
        conv = self._ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "mol2")
        conv.ReadFile(ob_mol, str(pdb_filepath))
        conv.WriteFile(ob_mol, str(mol2_filepath))

        if self.need_antechamber:
            mol_complex = self._molecule(str(mol2_filepath))
            ligand = mol_complex.copy()
            ligand.filter("resname UNL")
            ligand_file_path = f"{self.outdir}/{pdb_filepath.stem}_ligand.mol2"
            ligand.write(ligand_file_path)

            for atom_type in self.protein_atom_type:
                if atom_type in self.at:
                    antechamber_filepath = self.run_antechamber(ligand_file_path, atom_type)
                    mol_complex = self.replace_atom_type(mol_complex, antechamber_filepath)

            mol_complex.write(str(mol2_filepath))

        return str(mol2_filepath)

    def run_antechamber(self, filepath: Path, atom_type: str) -> str:
        antechamber_filepath = Path(f"{self.outdir}/{filepath.stem}_{self.at[atom_type]}.mol2")
        cmd_str = f"antechamber -i {str(filepath)} -fi mol2 -o {antechamber_filepath} -fo mol2 -at {self.at[atom_type]}"
        cmd = cmd_str.split(" ")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise FeaturizationError(f"Something went wrong with antechamber cmd: {cmd_str}, error: {error_msg}")

        return str(antechamber_filepath)

    def replace_atom_type(self, mol_complex, antechamber_filepath: str):
        ligand = self._molecule(antechamber_filepath)
        for atom_id in range(ligand.numAtoms):
            atom_name = ligand.name[atom_id]
            res_name = ligand.resname[atom_id]
            res_id = ligand.resid[atom_id]
            chain = ligand.chain[atom_id]
            atom_type = ligand.atomtype[atom_id]

            mask = (
                (mol_complex.name == atom_name)
                & (mol_complex.resname == res_name)
                & (mol_complex.resid == res_id)
                & (mol_complex.chain == chain)
            )

            matching_indices = np.where(mask)[0]
            if len(matching_indices) == 1:
                mol_complex.atomtype[matching_indices[0]] = atom_type
            else:
                raise FeaturizationError("Error in the update of the atom type")
        return mol_complex


class IntdescFeaturizer(AbstractFeaturizer):
    int_desc_location = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/"))

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        dir_to_save: str = None,
        distance: float = 8.0,
        ligand_residue: str = "UNL",
        solvent_name: str = "HOH",
        param_path: str = os.path.join(int_desc_location, "sample/ligand/param.yaml"),
        vdw_radius_path: str = os.path.join(int_desc_location, "sample/ligand/vdw_radius.yaml"),
        priority_path: str = os.path.join(int_desc_location, "sample/ligand/priority.yaml"),
        water_definition_path: str = os.path.join(int_desc_location, "water_definition.txt"),
        interaction_group_path: str = os.path.join(int_desc_location, "group.yaml"),
        allow_mediate_pos: Union[None, int] = None,
        on_14: bool = False,
        dup: bool = False,
        no_mediate: bool = False,
        no_out_total: bool = False,
        no_out_pml: bool = False,
        switch_ch_pi: bool = False,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        """
        There are 5 outputs generated in this featurizer:
        atom_ids, coords, protein_mask, edge_index, edge_features
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        from moleculekit.molecule import Molecule
        from ..vendor.riken.intDesc.interaction_descriptor import calculate

        self._molecule = Molecule
        self._intdesc_calculate = calculate

        self.outdir = Path(dir_to_save) if dir_to_save is not None else Path("/tmp/kmol/intdesc")
        self.distance = distance
        self.ligand_res = ligand_residue
        self._solvent_name = solvent_name
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
        # fmt: off
        self.interaction_labels = [
            "LP_CH_F","LP_CH_Hal_Br","LP_CH_Hal_Cl","LP_CH_Hal_I","LP_CH_N","LP_CH_O","LP_CH_PI","LP_CH_S","LP_Ca_X","LP_Cl_X","LP_Dipo","LP_Elec_NH_N","LP_Elec_NH_O","LP_Elec_OH_N","LP_Elec_OH_O","LP_Fe_X","LP_HB_NH_N","LP_HB_NH_O","LP_HB_OH_N","LP_HB_OH_O","LP_Hal_Br_N","LP_Hal_Br_O","LP_Hal_Br_S","LP_Hal_Cl_N","LP_Hal_Cl_O","LP_Hal_Cl_S","LP_Hal_I_N","LP_Hal_I_O","LP_Hal_I_S","LP_Hal_PI_Br","LP_Hal_PI_Cl","LP_Hal_PI_I","LP_K_X","LP_Mg_X","LP_NH_F","LP_NH_Hal_Br","LP_NH_Hal_Cl","LP_NH_Hal_I","LP_NH_PI","LP_NH_S","LP_Na_X","LP_Ni_X","LP_OH_F","LP_OH_Hal_Br","LP_OH_Hal_Cl","LP_OH_Hal_I","LP_OH_PI","LP_OH_S","LP_OMulPol","LP_PI_PI","LP_Zn_X","LP_vdW"
        ]
        # fmt: on

    def _process(self, data: Any, entry: DataPoint) -> Any:
        mol2_filepath = Path(data)

        mol_complex = self._molecule(str(mol2_filepath))
        uniq_resid = np.unique(
            mol_complex.get(
                "resid",
                sel=f"protein and not (resname {self.ligand_res}) and within {self.distance} of (resname {self.ligand_res} and noh)",
            )
        )
        # Write out yaml configuration
        param = {
            "ligand": {"name": self.ligand_res},
            "protein": {"num": uniq_resid.tolist()},
            "solvent": {"name": self._solvent_name},
        }

        molcular_select_filepath = Path(f"{self.outdir}/{mol2_filepath.stem}/molecular_select.yaml")

        with open(molcular_select_filepath, "w") as file:
            yaml.dump(param, file, indent=4)

        output_intdesc_dir = Path(f"{self.outdir}/{mol2_filepath.stem}/output_intdesc")

        # Run IntDesc
        self._intdesc_calculate(
            exec_type="Lig",
            mol2=str(mol2_filepath),
            molcular_select_file=str(molcular_select_filepath.absolute()),
            output=str(output_intdesc_dir),
            **self.intdesc_params,
        )

        intdesc = pd.read_csv(output_intdesc_dir + "_one_hot_list.csv")

        ligand_atom_ids = mol_complex.get("index", sel=f"resname {self.ligand_res}")
        protein_atom_ids = self.filter_protein_atom_of_interest(mol_complex, intdesc)

        mol_edge_index, edge_index = self.compute_edge_index(intdesc, ligand_atom_ids, protein_atom_ids)
        edge_features = self.compute_edge_features(intdesc, mol_edge_index, edge_index)

        coords = np.vstack([mol_complex.coords[ligand_atom_ids], mol_complex.coords[protein_atom_ids]])
        atomtype = np.hstack([mol_complex.atomtype[ligand_atom_ids], mol_complex.atomtype[protein_atom_ids]])
        protein_mask = np.hstack([np.zeros(len(ligand_atom_ids)), np.ones(len(protein_atom_ids))])

        return [atomtype, coords, protein_mask, edge_index, edge_features]

    def filter_protein_atom_of_interest(self, mol_complex, intdesc):
        """
        Recursively parse the protein graph from the lp interaction protein atom
        only adding protein atom in the cutoff distance.
        """
        protein_atom_in_cutoff = mol_complex.get(
            "index",
            sel=f"protein and not (resname {self.ligand_res}) and within {self.distance} of (resname {self.ligand_res} and noh)",
        )

        protein_atom_with_desc = np.unique(intdesc["partner_atom_number"].values)
        # Recursive computation
        neighbor_atoms = self._get_bonded_atom_in_selection(
            protein_atom_with_desc, protein_atom_in_cutoff, mol_complex.bonds
        )
        filtered_atom_ids = protein_atom_with_desc.tolist()
        while len(neighbor_atoms) != 0:
            neighbor_atoms = np.unique(neighbor_atoms[np.invert(np.isin(neighbor_atoms, filtered_atom_ids))])
            filtered_atom_ids += neighbor_atoms.tolist()
            neighbor_atoms = self._get_bonded_atom_in_selection(neighbor_atoms, protein_atom_in_cutoff, mol_complex.bonds)

        return filtered_atom_ids

    def compute_edge_index(self, intdesc, ligand_atom_ids, filter_protein_atom_ids):
        """
        The graph edge will use ligand first and protein for ordering ids.
        """
        all_atom = ligand_atom_ids + filter_protein_atom_ids
        graph_mol_index_mapping = {i: j for (i, j) in zip(range(len(all_atom)), all_atom)}

        mol_edge_index = np.unique(intdesc[["atom_number", "partner_atom_number"]].values, axis=0)

        edge_index = mol_edge_index.copy()
        for new_index, mol_index in graph_mol_index_mapping.items():
            edge_index[edge_index == mol_index] = new_index

        return mol_edge_index, edge_index

    def compute_edge_features(self, intdesc, mol_edge_index, edge_index):
        # [num_bonds, num_interaction_labels]
        edge_feature = np.zeros((edge_index.shape[0], len(self.interaction_labels)))

        for i, (atom_number, partner_atom_number) in enumerate(mol_edge_index):
            rows = intdesc[(intdesc.atom_number == atom_number) & (intdesc.partner_atom_number == partner_atom_number)]
            for j, interaction_label in enumerate(self.interaction_labels):
                if interaction_label in intdesc.columns:
                    edge_feature[i, j] = rows[interaction_label].sum()

        return edge_feature

    def _get_bonded_atom_in_selection(self, source_atom, selected_atoms, bond_id):
        """
        Return the selected_atoms bonded with source_atom.
        """
        where_contain_src_atoms = np.isin(bond_id[:, 0], source_atom) | np.isin(bond_id[:, 1], source_atom)
        dest_atoms = np.unique(bond_id[where_contain_src_atoms].ravel())
        dest_atoms = dest_atoms[np.invert(np.isin(dest_atoms, source_atom))]
        selected_dest_atom = dest_atoms[np.isin(dest_atoms, selected_atoms)]
        return selected_dest_atom
