import os
import itertools
import pyximport
from abc import ABCMeta, abstractmethod
from functools import lru_cache, partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import random
import numpy as np
import torch
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
from ..model.architectures import MsaExtractor
from .loaders import ListLoader

pyximport.install(setup_args={"include_dirs": np.get_include()})
from ..vendor.graphormer import algos  # noqa: E402


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
        self, inputs: List[str], outputs: List[str], classes: List[str], should_cache: bool = False, rewrite: bool = True
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
    def __init__(self, inputs: List[str], outputs: List[str], value: float, should_cache: bool = False, rewrite: bool = True):
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
        @param msa_extrator_cfg: Config for MsaExtractor model.
        """

        super().__init__(inputs, outputs, should_cache, rewrite)

        self.msa_extrator_cfg = msa_extrator_cfg
        if msa_extrator_cfg is not None:
            assert not msa_extrator_cfg.get("finetune", False), "In featurizer mode the extractor can't be finetune"
            self.msa_extrator = MsaExtractor(**msa_extrator_cfg)
            self.msa_extrator.eval()

        if msa_extrator_cfg is not None:
            self.config = self.msa_extrator.cfg
        else:
            self.config = model_config(name_config)

        self.config.data.predict.crop_size = crop_size

        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir, max_template_date="2022-11-03", max_hits=4, kalign_binary_path=""
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
            features = []
            with torch.no_grad():
                for i in range(processed_feature_dict["aatype"].shape[-1]):
                    fetch_cur_batch = lambda t: t[..., i].unsqueeze(0)  # noqa: E731
                    features.append(self.msa_extrator(tensor_tree_map(fetch_cur_batch, processed_feature_dict)))
                return features
        return processed_feature_dict

    def compute_unique_dataset_to_cache(self, loader: ListLoader):
        """
        Return a unique set of protein to cache in the featurizer.
        """
        data_source = loader._dataset
        data_source.pop("index")
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
