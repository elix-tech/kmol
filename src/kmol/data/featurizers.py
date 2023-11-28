import os
import io
import itertools
from pathlib import Path
import pyximport
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import pickle
import yaml
import subprocess

from contextlib import redirect_stdout
from io import StringIO


import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.data import Data as TorchGeometricData
from scipy.stats import bernoulli
import logging

from moleculekit.molecule import Molecule
from openbabel import openbabel, pybel

from kmol.data.resources import DataPoint
from .resources import DataPoint
from ..core.exceptions import FeaturizationError
from ..core.helpers import SuperFactory
from ..core.logger import LOGGER as logger
from ..vendor.openfold.data import templates, data_pipeline, feature_pipeline
from ..vendor.openfold.config import model_config
from ..vendor.openfold.utils.tensor_utils import tensor_tree_map
from ..model.architectures import AlphaFold
from ..vendor.riken.intDesc.interaction_descriptor import calculate
from .loaders import ListLoader

import algos

[l.setLevel(logging.WARNING) for l in logging.root.handlers]
logging.getLogger("kmol.vendor.riken.intDesc").setLevel(logging.ERROR)


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
                data.inputs[self._outputs[index]] = self.__process(raw_data, data)
                if self._rewrite:
                    data.inputs.pop(self._inputs[index])
        except (
            FeaturizationError,
            ValueError,
            IndexError,
            AttributeError,
            TypeError,
        ) as e:
            error_msg = "[WARNING] Could not run featurizer '{}' on '{}' --- {}".format(self.__class__.__name__, data.id_, e)
            raise FeaturizationError(error_msg) from e


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
        tokenize_onehot: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._vocabulary = vocabulary
        self._separator = separator
        self._max_length = max_length
        self._tokenize_onehot = tokenize_onehot

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        tokens = data.split(self._separator) if self._separator else [character for character in data]
        features = np.zeros((self._max_length, len(self._vocabulary)))

        for index, token in enumerate(tokens):
            if index == self._max_length:
                logger.warning("[CAUTION] Input is out of bounds. Features will be trimmed. --- {}".format(data))
                break

            features[index][self._vocabulary.index(token)] = 1
        if self._tokenize_onehot:
            features = features.argmax(1)

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

class IndexFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        vocabulary: List[str],
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)
        self._vocabulary = vocabulary
        self._to_index_dict = {v: k for k, v in enumerate(vocabulary)}

    def _process(self, data: str, entry: DataPoint) -> torch.FloatTensor:
        sample = [self._to_index_dict[amino_acid] for amino_acid in data]
        return torch.LongTensor(sample)

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

    def _process(self, data: str, entry: DataPoint) -> str:
        return pybel.readstring(self._source_format, data).write(self._target_format).strip()


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
    """Convert a pdbqt file to pdb"""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        pdqt_dir: str = None,
        dir_to_save: str = "data/output/pdb",
        overwrite_if_exist: bool = True,
        protonize: bool = False,
        ph: float = 7.0,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        """
        inputs: expect a column name of a pdbqt path data
        pdqt_dir: path to add to the input path field of the dataset.
        dir_to_save: pdb file are save to this directory
        protonize: Protonize if true
        ph: Use for protonization
        overwrite_if_exist: if False use existing files
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        self.outdir = Path(dir_to_save)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        self.pdqt_dir = pdqt_dir
        self.protonize = protonize
        self.ph = ph
        self.overwrite_if_exist = overwrite_if_exist

    def _process(self, data: str, entry: DataPoint) -> str:
        if self.pdqt_dir is not None:
            pdbqt_filepath = Path(f"{self.pdqt_dir}/{data}")
        else:
            pdbqt_filepath = Path(data)
        pdb_filepath = Path(f"{self.outdir}/{pdbqt_filepath.stem}.pdb")
        if not self.overwrite_if_exist and pdb_filepath.exists():
            return str(pdb_filepath)
        # Convert PDBQT to PDB
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("pdbqt", "pdb")
        ob_mol = openbabel.OBMol()
        # Avoid warning for kekulize aromatic bonds
        openbabel.obErrorLog.SetOutputLevel(0)
        if not conv.ReadFile(ob_mol, str(pdbqt_filepath)):
            raise FeaturizationError(f"Error while reading the pdbqt file for {str(pdbqt_filepath)}")
        openbabel.obErrorLog.SetOutputLevel(1)

        if self.protonize:
            ob_mol.DeleteHydrogens()
            res_name = [atom.GetResidue().GetName() for atom in openbabel.OBMolAtomIter(ob_mol)]
            ob_mol.CorrectForPH(self.ph)
            [atom.GetResidue().SetName(res_name) for atom, res_name in zip(openbabel.OBMolAtomIter(ob_mol), res_name)]
            ob_mol.AddHydrogens()

        if conv.WriteFile(ob_mol, str(pdb_filepath)):
            return str(pdb_filepath)
        else:
            raise FeaturizationError(f"Error while writing the pdb file for {str(pdbqt_filepath)}")


class PdbToMol2Featurizer(AbstractFeaturizer):
    """Convert a pdb file to mol2"""

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        pdb_dir: str = None,
        dir_to_save: str = "data/output/mol2",
        overwrite_if_exist: bool = True,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        """
        inputs: expect a column name of a pdb path data
        pdb_dir: path to add to the input taken from the dataset.
            Only use in case we are starting the featurization from a pdb files,
            otherwise leave as default
        overwrite_if_exist: if False use existing files, default: True
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        self.outdir = Path(dir_to_save)
        self.pdb_dir = pdb_dir
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        self.overwrite_if_exist = overwrite_if_exist

    def get_mol2_filepath(self, pdb_filepath):
        return Path(f"{self.outdir}/{Path(pdb_filepath).stem}.mol2")

    def pdb_to_mol2(self, pdb_filepath: str) -> str:
        mol2_filepath = self.get_mol2_filepath(pdb_filepath)
        if not self.overwrite_if_exist and mol2_filepath.exists():
            return str(mol2_filepath)
        ob_mol = openbabel.OBMol()
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("pdb", "mol2")
        if not conv.ReadFile(ob_mol, str(pdb_filepath)):
            raise FeaturizationError(f"Error while reading {str(pdb_filepath)} in OpenBabel")
        if not conv.WriteFile(ob_mol, str(mol2_filepath)):
            raise FeaturizationError(f"Error while writing {str(mol2_filepath)} in OpenBabel")

        return str(mol2_filepath)

    def get_filepath(self, data: Any):
        if self.pdb_dir is not None:
            return Path(self.pdb_dir) / data
        else:
            return data

    def _process(self, data: Any, entry: DataPoint) -> str:
        return self.pdb_to_mol2(self.get_filepath(data))


class AtomTypeExtensionPdbFeaturizer(PdbToMol2Featurizer):
    """
    From a pdb file generate two files a `.mol2` file generated with openbabel
    without any change in the atom type. And a second file `.moleculekit.mol2`
    with atom types modified.
    Since saving and loading a mol2 file with MoleculeKit make it lose the residue
    information, there is a need for saving those 2 different file.
    If multiple atom type are used they will be separated with a `,`.
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        pdb_dir: str = None,
        ligand_residue: List[str] = ["UNL"],
        protein_atom_type: List[str] = ["SYBYL"],
        ligand_atom_type: List[str] = ["SYBYL"],
        tokenize_atom_type: bool = True,
        dir_to_save: str = "data/output/antechamber",
        overwrite_if_exist: bool = True,
        should_cache: bool = False,
        rewrite: bool = True,
    ):
        """
        inputs: expect a column name of a pdb path data
        pdb_dir: path to add to the input taken from the dataset.
            Only use in case we are starting the featurization from a pdb files,
            otherwise leave as default
        ligand_residue: List of residue name of the ligand in the pdb file
        protein_atom_type: list of atom type to use option "SYBYL" and/or "PDB".
        ligand_atom_type: list of atom type to use option "SYBYL", "AM1-BCC" and/or "GAFF".
        dir_to_save: where to save the mol2 file and antechamber generation.
        tokenize_atom_type: Turn atom type to integer base on each atom type vocabulary.
        overwrite_if_exist: if False use existing files, default: True
        """
        super().__init__(inputs, outputs, pdb_dir, dir_to_save, overwrite_if_exist, should_cache, rewrite)

        self.ligand_residue = ligand_residue
        self.protein_atom_type = [s.upper() for s in protein_atom_type]
        self.ligand_atom_type = [s.upper() for s in ligand_atom_type]
        self.at = {"AM1-BCC": "bcc", "GAFF": "gaff"}
        self.need_antechamber = any([at in self.at for at in ligand_atom_type])
        self.tokenize_atom_type = tokenize_atom_type

        self.ligand_filter = f"resname {' '.join(self.ligand_residue)}"
        self.protein_filter = f"protein and not resname {' '.join(self.ligand_residue)}"

        # fmt: off
        # Taken from https://github.com/choderalab/ambermini/tree/master/share/amber/dat/antechamber + some missing one C.ar, C.3. H, cf
        self.additional_vocabulary = ['Al', 'Any', 'Br', 'C', 'C*', 'C.1', 'C.2', 'C.3', 'C.ar', 'C.cat', 'C1', 'CA', 'CB', 'CC', 'CD', 'CK', 'CM', 'CN', 'CQ', 'CR', 'CT', 'CV', 'CW', 'CY', 'CZ', 'Ca', 'Cl', 'DU', 'Du', 'F', 'H', 'H', 'H.spc', 'H.t3p', 'H1', 'H2', 'H3', 'H4', 'H5', 'HA', 'HC', 'HO', 'HP', 'HS', 'Hal', 'Het', 'Hev', 'I', 'K', 'LP', 'Li', 'N', 'N*', 'N.1', 'N.2', 'N.3', 'N.4', 'N.ar', 'N.pl3', 'N1', 'N2', 'N3', 'NA', 'NB', 'NC', 'NT', 'NY', 'Na', 'O', 'O.2', 'O.3', 'O.co2', 'O.spc', 'O.t3p', 'O2', 'OH', 'OS', 'OW', 'P', 'P.3', 'S', 'S.2', 'S.3', 'S.O', 'S.O2', 'SH', 'SO', 'Si', 'br', 'c', 'c1', 'c2', 'c3', 'ca', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cl', 'cp', 'cq', 'cu', 'cv', 'cx', 'cy', 'cz', 'du', 'f', 'h1', 'h2', 'h3', 'h4', 'h5', 'ha', 'hc', 'hn', 'ho', 'hp', 'hs', 'hw', 'hx', 'i', 'n', 'n', 'n1', 'n2', 'n3', 'n4', 'n7', 'n8', 'n9', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'nh', 'no', 'ns', 'nt', 'nu', 'nv', 'nx', 'ny', 'nz', 'o', 'oh', 'os', 'ow', 'p2', 'p3', 'p4', 'p5', 'pb', 'pc', 'pd', 'pe', 'pf', 'px', 'py', 's', 's2', 's4', 's6', 'sh', 'ss', 'sx', 'sy']
        self.gaff_vocabulary = ["Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Ca", "Cd", "Ce", "Cf", "Cm", "Co", "Cr", "Cs", "Cu", "C.ar", "C.3", "DU", "Db", "Ds", "Dy", "Er", "Es", "Eu", "Fe", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg", "Ho", "Hs", "In", "Ir", "K", "Kr", "LP", "La", "Li", "Lr", "Lu", "Md", "Mg", "Mn", "Mo", "Mt", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np", "Os", "O.3", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rh", "Rn", "Ru", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr", "br", "c", "c1", "c2", "c3", "ca", "cc", "ce", "cf", "cg", "ch", "cl", "cp", "cu", "cv", "cx", "cy", "cz", "f", "h1", "h2", "h3", "h4", "h5", "ha", "hc", "hn", "ho", "hp", "hs", "hw", "hx", "i", "lp", "n", "n1", "n2", "n3", "n4", "na", "nb", "nc", "ne", "nh", "no", "o", "oh", "os", "p2", "p3", "p4", "p5", "pb", "pc", "pe", "px", "py", "s", "s2", "s4", "s6", "sh", "ss", "sx", "sy"]
        self.bcc_vocabulary = {v: k for k,v in enumerate(["11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", "31", "32", "33", "42", "41", "51", "52", "53", "61", "71", "72", "73", "74", "91"])}
        self.pdb_vocabulary = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Th": 90, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}
        self.at_vocabulary = {
            "AM1-BCC": self.bcc_vocabulary,
            "GAFF": {v: k for k,v in enumerate(np.unique(self.gaff_vocabulary + self.additional_vocabulary))},
            "PDB": self.pdb_vocabulary
        }
        # Taken from http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
        self.sybyl_vocabulary = {v: k for k,v in enumerate(["C.3","C.2","C.1","C.ar","C.cat","N.3","N.2","N.1","N.ar","N.am","N.pl3","N.4","Na","O.3","O.2","O.co2","O.spc","O.t3p","S.3","S.2","S.O","S.O2","P.3","F","H","H.spc","H.t3p","LP","Du","Du.C","Any","Hal","Het","Hev","Li","Na","Mg","Al","Si","K","Ca","Cr.th","Cr.oh","Mn","Fe","Co.oh","Cu","Cl","Br","I","Zn","Se","Mo","Sn"])}
        # fmt: on

        self.get_atom_type_func = {"PDB": self.get_pdb_atom_type}

    def _process(self, data: Any, entry: DataPoint) -> Any:
        pdb_filepath = self.get_filepath(data)
        mol2_filepath = self.get_mol2_filepath(pdb_filepath)
        do_overwrite = self.overwrite_if_exist and Path(mol2_filepath).exists()
        if do_overwrite or not Path(mol2_filepath).exists():
            self.pdb_to_mol2(pdb_filepath)
        if not do_overwrite and Path(mol2_filepath).with_suffix(".tokenize_at.mol2").exists():
            return mol2_filepath
        try:
            mol_complex = Molecule(str(mol2_filepath))
        except Exception as e:
            raise FeaturizationError(f"Error while reading {str(mol2_filepath)} with MoleculeKit") from e

        self.compute_gaff_or_bcc_atom_type(mol_complex, pdb_filepath)

        # Update ligand atom type
        overwrite = "SYBYL" not in self.ligand_atom_type
        if self.tokenize_atom_type:
            mol_complex.atomtype = [str(self.sybyl_vocabulary[z]) for z in mol_complex.atomtype]
            mol_complex.atomtype = np.array(mol_complex.atomtype, dtype="object")
        for i, atom_type in enumerate(self.ligand_atom_type):
            if "SYBYL" == atom_type:
                continue
            atom_types = self.get_antechamber_atom_type(pdb_filepath, atom_type)
            mol_complex = self.update_atom_type(atom_type, mol_complex, atom_types, self.ligand_filter, overwrite)

        # Update protein atom type
        overwrite = "SYBYL" not in self.protein_atom_type
        if "PDB" in self.protein_atom_type:
            atom_types = self.get_pdb_atom_type(mol_complex)
            mol_complex = self.update_atom_type("PDB", mol_complex, atom_types, self.protein_filter, overwrite)

        # Set atom type on the openbabel object to have a faster write.
        ob_mol = openbabel.OBMol()
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("mol2", "mol2")
        if not conv.ReadFile(ob_mol, str(mol2_filepath)):
            raise FeaturizationError(f"Error while reading {str(pdb_filepath)} in OpenBabel")
        [atom.SetType(at) for atom, at in zip(openbabel.OBMolAtomIter(ob_mol), mol_complex.atomtype)]
        openbabel.obErrorLog.SetOutputLevel(0)
        if not conv.WriteFile(ob_mol, str(Path(mol2_filepath).with_suffix(".tokenize_at.mol2"))):
            raise FeaturizationError(f"Error while writing {str(mol2_filepath)} in OpenBabel")
        openbabel.obErrorLog.SetOutputLevel(1)
        return mol2_filepath

    def compute_gaff_or_bcc_atom_type(self, mol_complex: Molecule, pdb_filepath: str):
        """
        Run antechamber for each atom type needed, three files will be generated:
            - the input of antechamber `{name}_ligand.pdb` file containing only the ligand atoms
            - the output of antechamber `{name}_ligand_{atom_type}.mol2`
            - the a complex file where the ligand type have been updated `{name}_{atom_type}.mol2`
        """
        if self.need_antechamber:
            with redirect_stdout(StringIO()):
                ligand_file_path = self.create_ligand_pdb_file(pdb_filepath)
            for atom_type in self.ligand_atom_type:
                if atom_type in self.at:
                    mol_antechamber = self.run_antechamber(ligand_file_path, atom_type, mol_complex)
                    mol_antechamber.write(self.get_antechamber_filepath(pdb_filepath, atom_type))

    def run_antechamber(self, ligand_filepath_pdb: str, atom_type: str, mol_complex: Molecule) -> Molecule:
        """
        Run antechamber between the ligand pdb file and generate a mol2 file with the atom_type provided
        """
        antechamber_filepath = self.get_antechamber_filepath(ligand_filepath_pdb, atom_type)
        cmd_str = f"antechamber -i {ligand_filepath_pdb} -fi pdb -o {antechamber_filepath} -fo mol2 -at {self.at[atom_type]}"
        cmd = cmd_str.split(" ")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise FeaturizationError(f"Something went wrong with antechamber cmd: {cmd_str}, error: {error_msg}")
        mol_complex = self.replace_atom_type(mol_complex, antechamber_filepath)
        return mol_complex

    def get_antechamber_filepath(self, filepath_pdb: str, atom_type: str):
        return str(Path(self.outdir) / f"{Path(filepath_pdb).stem}_{self.at[atom_type]}.mol2")

    def create_ligand_pdb_file(self, pdb_filepath: Path) -> str:
        """
        Retrieve all lines starting with ATOM or HETATM if the residue is in `ligand_residue`
        and generate a new pdb file only with those atoms.
        Delete all CONECT.
        """
        ligand_file_path = Path(self.outdir) / f"{Path(pdb_filepath).stem}_ligand.pdb"
        atom_id = 1
        atom_id_space = " " * 5
        with open(str(pdb_filepath), "r") as infile, open(ligand_file_path, "w") as outfile:
            for line in infile:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if np.any([res in line for res in self.ligand_residue]):
                        line = line[:6] + atom_id_space[: -len(str(atom_id))] + str(atom_id) + line[11:]
                        outfile.write(line)
                        atom_id += 1
                elif not line.startswith("CONECT"):
                    outfile.write(line)
        return str(ligand_file_path)

    def replace_atom_type(self, mol: Molecule, antechamber_filepath: str):
        """
        Replace the atom type from the antechamber_filepath in the mol object.
        """
        mol_complex = mol.copy()
        ligand = Molecule(antechamber_filepath)
        for atom_id in range(ligand.numAtoms):
            res_name = ligand.resname[atom_id]
            res_id = ligand.resid[atom_id]
            chain = ligand.chain[atom_id]
            coords = ligand.coords[atom_id].reshape(-1, 1)
            atom_type = ligand.atomtype[atom_id]

            mask = (
                (mol_complex.resname == res_name)
                & (mol_complex.resid == res_id)
                & (mol_complex.chain == chain)
                & ((mol_complex.coords == coords).all(axis=1)[:, 0])
            )

            matching_indices = np.where(mask)[0]
            if len(matching_indices) == 1:
                mol_complex.atomtype[matching_indices[0]] = atom_type
            else:
                raise FeaturizationError(f"Error in the update of the atom type {antechamber_filepath}")
        return mol_complex

    def update_atom_type(self, atom_type, mol_complex: Molecule, additional_atom_type, sel, overwrite):
        """
        Update the main Molecule object with the additional_atom_type and tokenize them if needed
        """
        index = mol_complex.get("index", sel=sel)
        if overwrite:
            mol_complex.atomtype[index] = additional_atom_type
        else:
            if self.tokenize_atom_type:
                link_atom = lambda a, b: f"{a},{self.at_vocabulary[atom_type][b]}"
            else:
                link_atom = lambda a, b: f"{a},{b}"
            updated_atom_type = [link_atom(a, b) for a, b in zip(mol_complex.atomtype[index], additional_atom_type)]
            mol_complex.atomtype[index] = np.array(updated_atom_type)
        return mol_complex

    def get_pdb_atom_type(self, mol_complex):
        atom_type = mol_complex.get("element", sel=self.protein_filter)
        return atom_type

    def get_antechamber_atom_type(self, pdb_filepath: str, atom_type: str):
        filepath = self.get_antechamber_filepath(pdb_filepath, atom_type)
        new_atom_type = Molecule(filepath).get("atomtype", sel=self.ligand_filter)
        return new_atom_type


class IntdescFeaturizer(AbstractFeaturizer):
    """
    Run the interaction descriptor (IntDesc) program on a mol2 file and generate
    all inputs necessary for the Schnet Model.
    It needs a .moleculekit.mol2 file as well with the same path as the mol2 file.
    """

    int_desc_location = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/"))

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        dir_to_save: str = None,
        distance: float = 8.0,
        ligand_residue: List[str] = ["UNL"],
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
        There are 6 outputs generated and used in the model in this featurizer.
        They are regroup in a pytorch geometric Data object:

        z: Tokenize ligand atom type [N, num_atom_type]
        z_protein: Tokenize protein atom type [N, num_atom_type]
        edge_index: Edge index of the protein-ligand interaction
        edge_attr: Features of the protein-ligand interaction
        coords: Coordinates of both ligand and protein atoms, for each sample,
            ligand are first along the batch dimension
        protein_mask: mask marking protein atoms
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        self.outdir = Path(dir_to_save) if dir_to_save is not None else Path("data/output/intdesc")
        self.distance = distance
        self.ligand_res = ligand_residue
        self.ligand_filter = " ".join(self.ligand_res)
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

    def is_of_interest(self, protein_atom, ligand_atoms):
        return np.min([lig.GetDistance(protein_atom) for lig in ligand_atoms]) < self.distance

    def _process(self, data: Any, entry: DataPoint) -> Any:
        mol2_filepath = Path(data)

        mol_complex = Molecule(str(mol2_filepath), validateElements=False)
        uniq_resid = np.unique(
            mol_complex.get(
                "resid",
                sel=f"protein and not (resname {self.ligand_filter}) and within {self.distance} of (resname {self.ligand_filter} and noh)",
            )
        )
        # Write out yaml configuration
        ligand_name = np.unique(mol_complex.get("resname", sel=f"resname {self.ligand_filter}")).tolist()
        if len(ligand_name) != 1:
            raise FeaturizationError(
                f"The file {mol2_filepath} has {len(ligand_name)} (should be 1) ligand present list of ligand provided: {self.ligand_res}"
            )
        if len(uniq_resid) == 0:
            raise FeaturizationError(
                f"The file {mol2_filepath} has no residue found for the distance {self.distance} and the ligand provided: {self.ligand_res}"
            )
        param = {
            "ligand": {"name": mol_complex.get("resname", sel=f"resname {self.ligand_filter}").tolist()[0]},
            "protein": {"num": uniq_resid.tolist()},
            "solvent": {"name": self._solvent_name},
        }

        molcular_select_filepath = Path(f"{self.outdir}/{mol2_filepath.stem}/molecular_select.yaml")
        molcular_select_filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(str(molcular_select_filepath), "w") as file:
            yaml.dump(param, file, indent=4)

        output_intdesc_prefix = Path(f"{self.outdir}/{mol2_filepath.stem}/output_intdesc")

        # Run IntDesc
        intdesc_output = StringIO()
        with redirect_stdout(intdesc_output):
            calculate(
                exec_type="Lig",
                mol2=str(mol2_filepath),
                molcular_select_file=str(molcular_select_filepath.absolute()),
                output=str(output_intdesc_prefix),
                **self.intdesc_params,
            )
        logger.debug(intdesc_output)
        ligand_atom_ids = mol_complex.get("index", sel=f"resname {self.ligand_filter}")
        try:
            intdesc = pd.read_csv(str(output_intdesc_prefix) + "_one_hot_list.csv")
            intdesc[["atom_number", "partner_atom_number"]] = intdesc[["atom_number", "partner_atom_number"]] - 1

        except pd.errors.EmptyDataError as e:
            raise FeaturizationError(
                f"No interaction between Ligand and Protein found (intDesc result are empty) in {mol2_filepath}"
            ) from e

        protein_atom_ids = self.filter_protein_atom_of_interest(mol_complex, intdesc)

        mol_edge_index, edge_index, graph_mol_index_mapping = self.compute_edge_index(
            intdesc, ligand_atom_ids.tolist(), protein_atom_ids
        )
        edge_features = self.compute_edge_features(intdesc, mol_edge_index, edge_index)
        coords = np.vstack([mol_complex.coords[ligand_atom_ids], mol_complex.coords[protein_atom_ids]]).squeeze()
        # Retrieve tokenized atom type, the order is the same so we can use the same indexes.
        raw_atom_types = Molecule(str(Path(mol2_filepath).with_suffix(".tokenize_at.mol2")), validateElements=False).atomtype
        ligand_atom_types = np.array([a.split(",") for a in raw_atom_types[ligand_atom_ids]]).astype("int")
        protein_atom_types = np.array([a.split(",") for a in raw_atom_types[protein_atom_ids]]).astype("int")
        protein_mask = np.hstack([np.zeros(len(ligand_atom_ids)), np.ones(len(protein_atom_ids))])

        return TorchGeometricData(
            z=torch.from_numpy(ligand_atom_types).long(),
            z_protein=torch.from_numpy(protein_atom_types).long(),
            edge_index=torch.from_numpy(edge_index).long().T,
            edge_attr=torch.from_numpy(edge_features).float(),
            coords=torch.from_numpy(coords),
            protein_mask=torch.from_numpy(protein_mask).bool(),
            num_nodes=len(protein_mask),  # avoids warning
            # metadata for interprettation
            original_atom_ids=torch.from_numpy(np.hstack([ligand_atom_ids, protein_atom_ids])),
            mol2_path=str(mol2_filepath),
        )

    def filter_protein_atom_of_interest(self, mol_complex: Molecule, intdesc: pd.DataFrame) -> List:
        """
        Recursively parse the protein graph from the lp interaction protein atom
        only adding protein atom in the cutoff distance.
        """
        protein_atom_in_cutoff = mol_complex.get(
            "index",
            sel=f"protein and not (resname {self.ligand_filter}) and within {self.distance} of (resname {self.ligand_filter} and noh)",
        )

        protein_atom_with_desc = np.unique(intdesc["partner_atom_number"].values)
        all_filtered_atoms = set(protein_atom_with_desc.tolist())
        atoms_to_check = all_filtered_atoms

        while atoms_to_check:
            # Get neighboring atoms
            neighbor_atoms = self._get_bonded_atom_in_selection(atoms_to_check, protein_atom_in_cutoff, mol_complex.bonds)

            # Get new atoms that haven't been processed yet
            new_atoms = neighbor_atoms - all_filtered_atoms
            all_filtered_atoms.update(new_atoms)
            atoms_to_check = new_atoms

        return list(all_filtered_atoms)

    def compute_edge_index(self, intdesc, ligand_atom_ids, filter_protein_atom_ids):
        """
        The graph edge will use ligand first and protein for ordering ids.
        """
        all_atom = ligand_atom_ids + filter_protein_atom_ids
        graph_mol_index_mapping = dict(zip(range(len(all_atom)), all_atom))

        mol_edge_index = np.unique(intdesc[["atom_number", "partner_atom_number"]].values, axis=0)

        edge_index = mol_edge_index.copy()
        for new_index, mol_index in graph_mol_index_mapping.items():
            edge_index[mol_edge_index == mol_index] = new_index

        return mol_edge_index, edge_index, graph_mol_index_mapping

    def compute_edge_features(self, intdesc, mol_edge_index, edge_index):
        # [num_bonds, num_interaction_labels]
        edge_feature = np.zeros((edge_index.shape[0], len(self.interaction_labels)))

        for i, (atom_number, partner_atom_number) in enumerate(mol_edge_index):
            rows = intdesc[(intdesc.atom_number == atom_number) & (intdesc.partner_atom_number == partner_atom_number)]
            for j, interaction_label in enumerate(self.interaction_labels):
                if interaction_label in intdesc.columns:
                    edge_feature[i, j] = rows[interaction_label].sum()

        return edge_feature

    def _get_bonded_atom_in_selection(self, source_atoms: Set, selected_atoms: Set, bond_id: np.ndarray) -> Set:
        """
        Return the selected_atoms bonded with source_atom.
        """
        rows_with_source_atoms = np.any(np.isin(bond_id, list(source_atoms)), axis=1)
        bonded_atoms = bond_id[rows_with_source_atoms]

        bonded_atoms_set = set(bonded_atoms.ravel())
        # Remove source_atoms from bonded_atoms_set
        bonded_atoms_set.difference_update(source_atoms)
        # Find the intersection between bonded_atoms_set and selected_atoms
        selected_dest_atom = bonded_atoms_set.intersection(selected_atoms)
        return selected_dest_atom
