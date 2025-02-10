import os
import io
import itertools
from pathlib import Path
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import pickle
import logging
import random
import shutil


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.data import Data as TorchGeometricData
from scipy.stats import bernoulli
from openbabel import openbabel, pybel
import prody
import torch

from kmol.data.resources import DataPoint, IntDescRunner, AntechamberRunner, PDBtoMol2Converter, Mol2Processor
from kmol.data.loaders import ListLoader
from kmol.core.exceptions import FeaturizationError
from kmol.core.helpers import SuperFactory
from kmol.core.logger import LOGGER as logger
from kmol.model.architectures import AlphaFold
from kmol.vendor.openfold.data import templates, data_pipeline, feature_pipeline
from kmol.vendor.openfold.config import model_config
from kmol.vendor.openfold.utils.tensor_utils import tensor_tree_map

import algos

[l.setLevel(logging.WARNING) for l in logging.root.handlers]
logging.getLogger("kmol.vendor.riken.intDesc").setLevel(logging.ERROR)
prody.LOGGER._setverbosity("error")


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
        # try:
        if len(self._inputs) != len(self._outputs):
            raise FeaturizationError("Inputs and mappings must have the same length.")

        for index in range(len(self._inputs)):
            raw_data = data.inputs[self._inputs[index]]
            data.inputs[self._outputs[index]] = self.__process(raw_data, data)
            if self._rewrite:
                data.inputs.pop(self._inputs[index])
        # except (
        #     FeaturizationError,
        #     ValueError,
        #     IndexError,
        #     AttributeError,
        #     TypeError,
        # ) as e:
        #     error_msg = "[WARNING] Could not run featurizer '{}' on '{}' --- {}".format(self.__class__.__name__, data.id_, e)
        #     raise FeaturizationError(error_msg) from e


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
    """
    There is 1613 feature generated with this computer
    """

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

    def _process(self, data: str, entry: DataPoint):
        mol = Chem.MolFromSmiles(data)
        molecule_features = self._descriptor_calculator.run(mol, entry)
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
        """
        Featurizer converting a string to a list of index in vocabulary.
        This is mainly used along the padded collater for RNNs with nn.embedding as first layer.
        Max lenght of the sequence in batch is computed in the collater.
        """
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


class IntdescFeaturizer(AbstractFeaturizer):

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        dir_to_save: str = None,
        distance: float = 8.0,
        ligand_residue: List[str] = ["UNL"],
        solvent_name: str = "HOH",
        overwrite_if_exist: bool = True,
        should_cache: bool = False,
        rewrite: bool = True,
        **kwargs,
    ):
        """
        Run the interaction descriptor (IntDesc) program on a mol2 file and generate
        all inputs necessary for the Schnet Model.

        There are 8 outputs generated and used in the model in this featurizer.
        They are regrouped in a pytorch geometric Data object:

        z: Tokenize ligand atom type based on pdb_vocab [N_ligand]
        z_protein: Tokenize protein atom type based on pdb_vocab [N_protein]
        edge_index: Edge index of the protein-ligand interaction
        edge_attr: Features of the protein-ligand interaction
        coords: Coordinates of both ligand and protein atoms, for each sample,
            ligand are first along the batch dimension [N_ligand + N_protein, 3]
        protein_mask: Mask marking protein atoms [N_ligand + N_protein]
        num_nodes: Number of nodes in the graph
        original_atom_ids: Original atom IDs from the PDB file
        pdb_path: Path to the PDB file
        mol2_path: Path to the MOL2 file
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        self.outdir = Path(dir_to_save) if dir_to_save is not None else Path("data/output/intdesc")
        self.distance = distance
        self.ligand_res = ligand_residue
        self.ligand_filter = " ".join(self.ligand_res)
        self.solvent_name = solvent_name

        self.intdesc_runner = IntDescRunner(**kwargs)
        self.overwrite_if_exist = overwrite_if_exist

        # fmt off
        self.pdb_token_lookup = np.vectorize(AntechamberRunner.pdb_vocabulary.get)
        # fmt on

    def is_of_interest(self, protein_atom, ligand_atoms):
        return np.min([lig.GetDistance(protein_atom) for lig in ligand_atoms]) < self.distance

    def _process(self, data: str, entry: DataPoint) -> Any:
        pdb_path = data
        mol2_filepath = self.create_mol2_from_pdb(pdb_path)

        # Create a prody structure from the pdb file and add the bond information from the mol2 file
        # The use of the mol2 file is to match the previous implementation with MoleculeNet
        complex_struct = prody.parsePDB(pdb_path)
        bonds = Mol2Processor.extract_bond_indices_from_file(mol2_filepath)
        complex_struct.setBonds(bonds)

        intdesc = self.compute_intdesc(pdb_path, mol2_filepath, complex_struct)

        protein_atom_ids = self.filter_protein_atom_of_interest(complex_struct, intdesc)
        ligand_atom_ids = complex_struct.select(f"resname {self.ligand_filter} and noh").getIndices().tolist()

        edge_index = self.compute_edge_index(intdesc, ligand_atom_ids, protein_atom_ids)
        edge_features = self.compute_edge_features(intdesc)

        coords = complex_struct.getCoords()[ligand_atom_ids + protein_atom_ids]

        # Compute atom type from pdb file
        raw_atom_types = complex_struct.getElements()
        ligand_atom_types = self.pdb_token_lookup(raw_atom_types[ligand_atom_ids]).astype("int")
        protein_atom_types = self.pdb_token_lookup(raw_atom_types[protein_atom_ids]).astype("int")
        protein_mask = np.hstack([np.zeros(len(ligand_atom_ids)), np.ones(len(protein_atom_ids))])

        return TorchGeometricData(
            z=torch.from_numpy(ligand_atom_types).long(),
            z_protein=torch.from_numpy(protein_atom_types).long(),
            edge_index=torch.from_numpy(edge_index).long().T,
            edge_attr=torch.from_numpy(edge_features).float(),
            coords=torch.from_numpy(coords).float(),
            protein_mask=torch.from_numpy(protein_mask).bool(),
            num_nodes=len(protein_mask),  # avoids warning
            # metadata for interprettation
            original_atom_ids=torch.from_numpy(np.hstack([ligand_atom_ids, protein_atom_ids])),
            pdb_path=str(pdb_path),
            mol2_path=str(mol2_filepath),
        )

    def compute_intdesc(self, pdb_path: str, mol2_filepath: str, complex_struct: prody.AtomGroup) -> pd.DataFrame:
        """
        Computes interaction descriptors for a given molecular complex.
        Parameters:
        pdb_path (str): Path to the PDB file of the molecular complex.
        mol2_filepath (str): Path to the MOL2 file of the ligand.
        complex_struct (prody.AtomGroup): Structure of the molecular complex.
        Returns:
        pd.DataFrame: Computed interaction descriptors.
        Raises:
        Exception: If an error occurs during the computation, the output directory is removed and the exception is raised.
        """

        outdir = self.outdir / Path(pdb_path).stem
        if self.overwrite_if_exist or not outdir.exists():
            try:
                intdesc = self.intdesc_runner.run(
                    mol2_filepath, complex_struct, self.ligand_filter, self.distance, self.solvent_name, outdir
                )
            except Exception as e:
                shutil.rmtree(outdir)
                raise e
        else:
            intdesc = self.intdesc_runner.extract_output(outdir)
        return intdesc

    def create_mol2_from_pdb(self, pdb_path: Union[str, Path]) -> Path:
        """
        Converts a PDB file to a MOL2 file. Depending on overwrite_if_exist if
        the file already exists it will not be overwritten.
        Parameters:
        pdb_path (str or Path): The path to the PDB file to be converted.
        Returns:
        Path: The path to the generated MOL2 file.
        """

        mol2_filepath = PDBtoMol2Converter.get_mol2_filepath(self.outdir / "mol2_files", pdb_path)
        if self.overwrite_if_exist or not mol2_filepath.exists():
            mol2_filepath = PDBtoMol2Converter.pdb_to_mol2(
                pdb_path, self.outdir / "mol2_files", overwrite=self.overwrite_if_exist
            )

        return mol2_filepath

    def filter_protein_atom_of_interest(self, complex_struct: prody.AtomGroup, intdesc: pd.DataFrame) -> List:
        """
        Recursively parse the protein graph from the lp interaction protein atom
        only adding protein atom in the cutoff distance.
        1- Take all protein atom with LP interaction from intDesc.
        2- Take the neighbor of those atoms, and keep them if the cutoff distance is respected.
        3- Repeat the previous operation until we don't have anymore atoms
        """
        # keeping now for matching test case
        protein_atom_in_cutoff = complex_struct.select(
            f"protein and (not resname {self.ligand_filter}) and (within {self.distance} of resname {self.ligand_filter}) and noh"
        )
        protein_atom_in_cutoff = protein_atom_in_cutoff.getIndices()

        protein_atom_with_desc = np.unique(intdesc["partner_atom_number"].values)
        all_filtered_atoms = set(protein_atom_with_desc.tolist())
        atoms_to_check = all_filtered_atoms

        while atoms_to_check:
            # Get neighboring atoms
            neighbor_atoms = self._get_bonded_atom_in_selection(
                atoms_to_check, protein_atom_in_cutoff, complex_struct._bonds
            )

            # Get new atoms that haven't been processed yet
            new_atoms = neighbor_atoms - all_filtered_atoms
            all_filtered_atoms.update(new_atoms)
            atoms_to_check = new_atoms

        return list(all_filtered_atoms)

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

    def compute_edge_index(self, intdesc: pd.DataFrame, ligand_atom_ids: List[int], protein_atom_ids: List[int]):
        """
        The graph edge will use ligand first and protein for ordering ids.

        return
        - intdesc_edge_index: edge index with atom ids based on the full pdb file
        - edge_index: reindexed edge index with only selected atoms
        """
        atom_ids = ligand_atom_ids + protein_atom_ids
        # graph_mol_index_mapping = dict(zip(range(len(atom_ids)), atom_ids))

        intdesc_edge_index = np.unique(intdesc[["atom_number", "partner_atom_number"]].values, axis=0)

        edge_index = intdesc_edge_index.copy()
        for new_idx, idx in enumerate(atom_ids):
            edge_index[intdesc_edge_index == idx] = new_idx

        return edge_index

    def compute_edge_features(self, intdesc: pd.DataFrame):
        """
        Compute edge features for a given interaction descriptor DataFrame.

        Parameters:
            intdesc (pd.DataFrame): A DataFrame containing interaction descriptors with columns:
                - 'atom_number': Atom numbers for one side of the interaction.
                - 'partner_atom_number': Atom numbers for the interacting partners.
                - Interaction labels (e.g., hydrogen bonding, van der Waals, etc.).

        Returns:
            np.ndarray: A 2D array of shape [num_edges, num_interaction_labels], where each row corresponds
                        to an edge (unique atom-partner pair) and each column corresponds to the sum of a
                        specific interaction label for that edge.
        """
        # Create unique edges and their corresponding indices
        intdesc_edge_index, edge_mapping = np.unique(
            intdesc[["atom_number", "partner_atom_number"]].values, axis=0, return_inverse=True
        )

        # Initialize edge features
        num_edges = intdesc_edge_index.shape[0]
        num_labels = len(IntDescRunner.interaction_labels)
        edge_feature = np.zeros((num_edges, num_labels))

        # Sum interaction label values for each edge
        for j, interaction_label in enumerate(IntDescRunner.interaction_labels):
            if interaction_label in intdesc.columns:
                edge_feature[:, j] = intdesc.groupby(edge_mapping)[interaction_label].sum().values

        return edge_feature


class AtomTypeExtensionFeaturizer(AbstractFeaturizer):
    """
    From the output of IntdescFeaturizer, overwrite the basic pdb atom indexes with
    more complex atom type indexes comming from antechamber.
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        ligand_residue: List[str] = ["UNL"],
        protein_atom_type: List[str] = ["SYBYL"],
        ligand_atom_type: List[str] = ["SYBYL"],
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

        Note if SYBYL is used we are using the atom type from the mol2 generated file
        by openbabel, this can lead to different result than antechamber, but is a lot faster.
        """
        super().__init__(inputs, outputs, should_cache, rewrite)

        self.dir_to_save = dir_to_save
        self.ligand_residue = ligand_residue
        self.protein_atom_types = [s.upper() for s in protein_atom_type]
        self.ligand_atom_types = [s.upper() for s in ligand_atom_type]
        self.overwrite_if_exist = overwrite_if_exist

        self.ligand_filter = " ".join(self.ligand_residue)

        self.antechamber_runner = AntechamberRunner()

    def _process(self, data: TorchGeometricData, entry: DataPoint) -> Any:
        pdb_path = data["pdb_path"]
        complex_struct = prody.parsePDB(pdb_path)

        ligand_at = self.compute_ligand_atom_types(pdb_path, complex_struct)

        ligand_ids, protein_ids = data.original_atom_ids[~data.protein_mask], data.original_atom_ids[data.protein_mask]

        # Compute  additional atom type for protein note we don't use antechamber due to computation time on a complex.
        protein_at = {}
        if "PDB" in self.protein_atom_types:
            pdb_atom_types = complex_struct.getElements()
            pdb_protein_at = self.antechamber_runner.tokenize_atom_types(pdb_atom_types, "PDB")
            protein_at["PDB"] = pdb_protein_at[protein_ids]

        self.extract_sybyl_atom_type(data["mol2_path"], ligand_at, ligand_ids, protein_at, protein_ids)

        # Concatenate in the proper order and pass to tensor.
        ligand_at = np.stack([ligand_at[k] for k in self.ligand_atom_types], axis=1)
        protein_at = np.stack([protein_at[k] for k in self.protein_atom_types], axis=1)
        data["z"] = torch.from_numpy(ligand_at).long()
        data["z_protein"] = torch.from_numpy(protein_at).long()

        data["ligand_at"] = self.ligand_atom_types
        data["protein_at"] = self.protein_atom_types

        return data

    def compute_ligand_atom_types(self, pdb_path: str, complex_struct: prody.AtomGroup) -> Dict[str, np.ndarray]:
        """
        Compute the atom types for the ligand in a given PDB file.
        Parameters:
        -----------
        pdb_path : str
            The file path to the PDB file containing the ligand structure.
        complex_struct : prody.AtomGroup
            The complex structure object containing the ligand.
        Returns:
        --------
        Dict[str, np.ndarray]
            A dictionary where keys are atom type identifiers and values are numpy arrays of atom indices.
            If no ligand atom types are specified, an empty dictionary is returned.
        """

        ligand_atom_types = set(self.ligand_atom_types) - set(["SYBYL"])
        if len(ligand_atom_types):
            ligand_pdb_filepath = self.file_preparation(pdb_path, complex_struct)
            ligand_at = self.antechamber_runner.get_multiple_tokenize_atom_types(
                ligand_pdb_filepath, ligand_atom_types, Path(self.dir_to_save) / Path(pdb_path).stem, self.overwrite_if_exist
            )
            # Remove hydrogen atoms
            ligand = prody.parsePDB(ligand_pdb_filepath)
            non_h_atoms = ligand.select("noh").getIndices()
            ligand_at = {k: v[non_h_atoms] for k, v in ligand_at.items()}
        else:
            ligand_at = {}
        return ligand_at

    def file_preparation(self, pdb_filepath: Path, complex_struct: prody.AtomGroup) -> str:
        """Remove element that make antechamber fail and remove bonds."""
        ligand = complex_struct.select(f"resname {self.ligand_filter}")
        ligand = ligand.copy()
        ligand.setChids("")
        ligand.setElements("")
        dir_to_save = Path(self.dir_to_save) / Path(pdb_filepath).stem
        dir_to_save.mkdir(parents=True, exist_ok=True)
        ligand_file_path = str(dir_to_save / f"{Path(pdb_filepath).stem}_ligand.pdb")
        prody.writePDB(ligand_file_path, ligand)
        return ligand_file_path

    def extract_sybyl_atom_type(self, path_mol2, ligand_at, ligand_ids, protein_at, protein_ids):
        """
        We use the already computed mol2 file to be efficient since using antechamber is slow.
        Mol2 files have SYBYL atom type as default.
        So the SYBYL atom type are computed with openbabel in a previous step and not antechamber.
        """
        if "SYBYL" in self.protein_atom_types or "SYBYL" in self.ligand_atom_types:
            sybyl_atom_types = np.array(Mol2Processor.extract_atom_types_from_file(path_mol2))

        if "SYBYL" in self.ligand_atom_types:
            ligand_at["SYBYL"] = self.antechamber_runner.tokenize_atom_types(sybyl_atom_types[ligand_ids], "SYBYL")

        if "SYBYL" in self.protein_atom_types:
            protein_at["SYBYL"] = self.antechamber_runner.tokenize_atom_types(sybyl_atom_types[protein_ids], "SYBYL")

        return ligand_at, protein_at


class ReactionFeaturizer(AbstractFeaturizer):

    MODE_FORWARD_PREDICTION = "forward_prediction"
    MODE_RETROSYNTHESIS = "retrosynthesis"

    def __init__(
            self, inputs: List[str], outputs: List[str], mode: str = MODE_RETROSYNTHESIS,
            fingerprint_size: int = 2048, radius: int = 2, use_chirality: bool = False,
            should_cache: bool = False, rewrite: bool = True
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        self._mode = mode
        self._fingerprint_size = fingerprint_size
        self._radius = radius
        self._use_chirality = use_chirality

    def _process(self, data: str, entry: DataPoint) -> torch.Tensor:
        reactants, agents, products = [components.split(".") for components in data.split(">")]
        targets = products if self._mode == ReactionFeaturizer.MODE_RETROSYNTHESIS else reactants

        fingerprints = [
            CircularFingerprintFeaturizer.generate_fingerprint(
                mol=Chem.MolFromSmiles(compound),
                fingerprint_size=self._fingerprint_size,
                radius=self._radius,
                use_chirality=self._use_chirality
            )
            for compound in targets
        ]

        return torch.FloatTensor(np.sum(fingerprints, axis=0, dtype=np.uint8))
