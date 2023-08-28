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

import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.data import Data as TorchGeometricData
from scipy.stats import bernoulli

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


class PdbqtFeaturizer(AbstractFeaturizer):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        inputs: List[str],
        outputs: List[str],
        should_cache: bool = False,
        rewrite: bool = True,
        distance: float = 8.0,
        solvent_name: str = "HOH",
        param_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/sample/ligand/param.yaml")
        ),
        vdw_radius_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/sample/ligand/vdw_radius.yaml")
        ),
        priority_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/sample/ligand/priority.yaml")
        ),
        water_definition_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/water_definition.txt")
        ),
        interaction_group_path: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "vendor/riken/intDesc/group.yaml")
        ),
        allow_mediate_pos: Union[None, int] = None,
        on_14: bool = False,
        dup: bool = False,
        no_mediate: bool = False,
        no_out_total: bool = False,
        no_out_pml: bool = False,
        switch_ch_pi: bool = False,
    ):
        super().__init__(inputs, outputs, should_cache, rewrite)

        from ..vendor.riken.intDesc.interaction_descriptor import calculate
        from openbabel import openbabel as ob, pybel
        from moleculekit.molecule import Molecule
        import subprocess

        self._input_path = input_path
        self._output_path = output_path
        self._distance = distance
        self._solvent_name = solvent_name
        self._param_path = param_path
        self._vdw_radius_path = vdw_radius_path
        self._priority_path = priority_path
        self._water_definition_path = water_definition_path
        self._interaction_group_path = interaction_group_path
        self._allow_mediate_pos = allow_mediate_pos
        self._on_14 = on_14
        self._dup = dup
        self._no_mediate = no_mediate
        self._no_out_total = no_out_total
        self._no_out_pml = no_out_pml
        self._switch_ch_pi = switch_ch_pi

        self._ob = ob
        self._pybel = pybel
        self._molecule = Molecule
        self._subprocess = subprocess
        self._intdesc_calculate = calculate

        output_path = Path(self._output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # default naming convention
        # PROTEIN-ID_drug_id-DRUG-ID_best_model.pdbqt
        # i.e Q60805-drug_id-503404_best_model.pdbqt

    def _get_bonded_atom_in_cutoff_distance(self, atom_id_of_interest, atom_ids_in_cutoff, bond_id):
        atom_id_bond_of_interest = bond_id[
            np.isin(bond_id[:, 0], atom_id_of_interest) | np.isin(bond_id[:, 1], atom_id_of_interest)
        ].ravel()
        new_atom_ids = atom_id_bond_of_interest[np.invert(np.isin(atom_id_bond_of_interest, atom_id_of_interest))]
        new_atom_ids_in_cutoff = new_atom_ids[np.isin(new_atom_ids, atom_ids_in_cutoff)]
        return new_atom_ids_in_cutoff

    def _process(self, data: Any, entry: DataPoint) -> torch.FloatTensor:
        pdbqt_filepath = Path(entry["path"]) if "path" in entry.inputs else None

        if pdbqt_filepath is None:
            if "target_accession" in entry.inputs and "drug_id" in entry.inputs:
                filename = f"{entry.inputs['target_accession']}-drug_id-{entry.inputs['drug_id']}_best_model"
                pdbqt_filepath = Path(f"{self._input_path}/{filename}.pdbqt")
            else:
                raise FeaturizationError("No target_accession or drug_id found in data")

        if not pdbqt_filepath.exists():
            raise FeaturizationError(f"File {pdbqt_filepath} does not exist")

        pdb_filepath = Path(f"{self._output_path}/{filename}.pdb")
        mol2_filepath = Path(f"{self._output_path}/{filename}.mol2")
        molcular_select_filepath = Path(f"{self._output_path}/{filename}_molecular_select.yaml")
        intdesc_filepath = Path(f"{self._output_path}/{filename}_intdesc")

        # Convert PDBQT to PDB
        conv = self._ob.OBConversion()
        conv.SetInAndOutFormats("pdbqt", "pdb")
        ob_mol = self._ob.OBMol()
        conv.ReadFile(ob_mol, str(pdbqt_filepath))
        conv.WriteFile(ob_mol, str(pdb_filepath))

        # Protonize the complex and convert it from pdb -> mol2
        ob_mol.CorrectForPH(7.0)
        ob_mol.AddHydrogens()
        conv.SetInAndOutFormats("pdb", "mol2")
        conv.WriteFile(ob_mol, str(mol2_filepath))

        # Generates the parameter for IntDesc
        mol = self._molecule(str(mol2_filepath))
        uniq_resid = np.unique(
            mol.get("resid", sel=f"protein and not (resname UNL) and within {self._distance} of (resname UNL and noh)")
        )
        # uniq_chain = np.unique(mol.get("chain", sel=f"protein and not (resname UNL) and within {self._distance} of (resname UNL and noh)"))

        protein = mol.copy()
        protein.deleteBonds("all")
        protein.filter("protein and not resname UNL")
        # protein.write("protein.pdb")

        ligand = mol.copy()
        protein.deleteBonds("all")
        ligand.filter("resname UNL")
        # ligand.write("ligand.pdb")

        # Write out yaml configuration
        param = {"ligand": {"name": "UNL"}, "protein": {"num": uniq_resid.tolist()}, "solvent": {"name": self._solvent_name}}
        with open(molcular_select_filepath, "w") as file:
            yaml.dump(param, file, indent=4)

        # Run IntDesc
        self._intdesc_calculate(
            exec_type="Lig",
            mol2=str(mol2_filepath),
            molcular_select_file=str(molcular_select_filepath),
            parametar_file=self._param_path,
            vdw_file=self._vdw_radius_path,
            priority_file=self._priority_path,
            water_definition_file=self._water_definition_path,
            interaction_group_file=self._interaction_group_path,
            output=str(intdesc_filepath),
            allow_mediate_position=self._allow_mediate_pos,
            on_14=self._on_14,
            dup=self._dup,
            no_mediate=self._no_mediate,
            no_out_total=self._no_out_total,
            no_out_pml=self._no_out_pml,
            switch_ch_pi=self._switch_ch_pi,
        )

        # Read IntDesc output
        lp_desc = pd.read_csv(str(intdesc_filepath) + "_one_hot_list.csv")

        # if we want to specify the coordinate of the atoms close to ligand
        atom_coordinates = mol.get("coords", sel=f"protein and within {self._distance} of (resname UNL and noh)")
        # if we want to specify atom id of the atoms close to ligand
        protein_atom_ids = mol.get(
            "index", sel=f"protein and not (resname UNL) and within {self._distance} of (resname UNL and noh) "
        )
        ligand_atom_ids = mol.get("index", sel="resname UNL")

        protein_atom_with_desc = np.unique(lp_desc["partner_atom_number"].values)

        neighbor_atoms = self._get_bonded_atom_in_cutoff_distance(protein_atom_with_desc, protein_atom_ids, mol.bonds)
        filter_protein_atom_ids = protein_atom_with_desc.tolist()
        while len(neighbor_atoms) != 0:
            neighbor_atoms = np.unique(neighbor_atoms[np.invert(np.isin(neighbor_atoms, filter_protein_atom_ids))])
            filter_protein_atom_ids += neighbor_atoms.tolist()
            neighbor_atoms = self._get_bonded_atom_in_cutoff_distance(neighbor_atoms, protein_atom_ids, protein.bonds)

        # Retrieve coordinates of protein and ligand
        filter_protein_atom_ids = sorted(filter_protein_atom_ids)
        graph_mol_index = {i: j for (i, j) in zip(range(len(filter_protein_atom_ids)), filter_protein_atom_ids)}
        protein_coordinate = mol.coords[filter_protein_atom_ids]

        ligand_atom_ids = sorted(ligand_atom_ids)
        graph_mol_index = graph_mol_index | {
            i: j
            for (i, j) in zip(
                range(len(filter_protein_atom_ids), len(filter_protein_atom_ids) + len(ligand_atom_ids)), ligand_atom_ids
            )
        }
        ligand_coordinate = mol.coords[ligand_atom_ids]

        # Protein Ligand features
        # edge index generation
        mol_lp_edge_index = lp_desc[["atom_number", "partner_atom_number"]].values
        mol_lp_edge_index = np.unique(mol_lp_edge_index, axis=0)
        graph_lp_edge_index = mol_lp_edge_index.copy()
        for graph_index, mol_index in graph_mol_index.items():
            graph_lp_edge_index[graph_lp_edge_index == mol_index] = graph_index

        # Retrieve all possible interaction between ligand and protein
        int_desc_sum = pd.read_csv(str(intdesc_filepath) + "_interaction_count_list.csv", header=None)
        interaction_labels = int_desc_sum[~int_desc_sum[0].str.contains("#S")][0].values
        for i in range(len(interaction_labels)):
            interaction_labels[i] = interaction_labels[i].replace("#Pro", "")
            interaction_labels[i] = interaction_labels[i].replace("#", "P_")

        # Create edge_features num_bonds x interaction_labels.
        edge_feature = np.zeros((mol_lp_edge_index.shape[0], len(interaction_labels)))
        for i, (atom_number, partner_atom_number) in enumerate(mol_lp_edge_index):
            for j, interaction_label in enumerate(interaction_labels):
                if interaction_label in lp_desc.columns:
                    edge_feature[i, j] = lp_desc[
                        (lp_desc.atom_number == atom_number) & (lp_desc.partner_atom_number == partner_atom_number)
                    ][interaction_label].sum()

        # Run antechamber and obtain features with subprocess

        return
