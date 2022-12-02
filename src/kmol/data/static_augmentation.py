from abc import ABCMeta, abstractmethod
from copy import deepcopy
from datetime import datetime
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.BRICS import BRICSDecompose
from torch_geometric.data import Data as PyG_Data
from torch.utils.data import Subset

from ..core.helpers import SuperFactory
from ..core.logger import LOGGER as logging
from ..core.utils import progress_bar
from .loaders import AbstractLoader
from .resources import DataPoint
from mila.factories import AbstractScript


class AbstractStaticAugmentation(metaclass=ABCMeta):
    def __init__(self, featurization_jobs: int):
        self.featurization_jobs = featurization_jobs
        self.aug_dataset: List = []
        self.splits: Dict = {}

    @abstractmethod
    def generate_augmented_data(self, loader: AbstractLoader):
        raise NotImplementedError

    @abstractmethod
    def generate_splits(self, splits: Dict):
        raise NotImplementedError

    @abstractmethod
    def get_aug_split_name(self, split_name, *args, kwargs):
        raise NotImplementedError

    def _get_chunks(self, _list, dataset=None):
        n_jobs = self.featurization_jobs
        chunk_size = len(_list) // n_jobs
        chunks = [_list[i : i + chunk_size] for i in range(0, len(_list), chunk_size)]
        if dataset is None:
            return chunks
        else:
            return [Subset(dataset, chunk) for chunk in chunks]


class MotifRemovalStaticAugmentation(AbstractStaticAugmentation):
    """
    Base logic taken form Auglichem https://baratilab.github.io/AugLiChem/molecule.html
    """

    def __init__(self, similarity_threshold=0.6, smiles_field="smiles", max_per_mol=-1, *args, **kwargs):
        """
        @param similarity_threshold: (float) The minimum tanimoto similarity for a molecule to be kept
        @param smiles_field: the name of the column input containing the smile information
        @param max_per_mol: set a maximum number of additional molecule per molecules.
        """
        super().__init__(*args, **kwargs)
        self.max_per_mol = max_per_mol
        self.similarity_threshold = similarity_threshold
        self.smiles_field = smiles_field

    # def apply_transform(self, mol: rdkit.Chem.rdchem.Mol) -> List[rdkit.Chem.rdchem.Mol]:
    def apply_transform(self, data: DataPoint) -> PyG_Data:
        """
        Transform that randomly remove a motif decomposed via BRICS
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @returns: list of augmented rdkit.Chem.rdchem.Mol
        """
        mol = data.inputs[self.smiles_field]
        if isinstance(mol, PyG_Data):
            mol = Chem.MolFromSmiles(mol.smiles)
        elif isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        augs_data = []
        fp_data = []
        fp = Chem.RDKFingerprint(mol)
        res = list(BRICSDecompose(mol, returnMols=False, singlePass=True))
        for r in res:
            mol_aug = Chem.MolFromSmiles(r)
            fp_aug = Chem.RDKFingerprint(mol_aug)
            if DataStructs.FingerprintSimilarity(fp, fp_aug) > self.similarity_threshold:
                data_aug = deepcopy(data)
                data_aug.id_ = -1
                data_aug.original_data_id = data.id_
                data_aug.inputs[self.smiles_field] = Chem.MolToSmiles(mol_aug)
                augs_data.append(data_aug)
                fp_data.append(fp_aug)

        if self.max_per_mol > 0 and len(augs_data) > self.max_per_mol:
            augs_data = self.filter_most_dissimilar_subset(augs_data, fp_data)

        return augs_data

    def filter_most_dissimilar_subset(self, augs_data, fp_data):
        filter_data = [augs_data[0]]
        last_selected_fp = fp_data[0]
        fp_tanimoto_sim = {fp: [] for fp in fp_data[1:]}
        fp_mol = {fp: aug for fp, aug in zip(fp_data[1:], augs_data[1:])}
        for i in range(self.max_per_mol - 1):
            # Compute similarity with last selected molecule
            for fp, v in fp_tanimoto_sim.items():
                fp_tanimoto_sim[fp].append(DataStructs.FingerprintSimilarity(fp, last_selected_fp))
            # Add the molecule with the average smallest tanimoto similarity
            # And remove it from the possible solution
            id_ = np.argmin([np.mean(v) for v in fp_tanimoto_sim.values()])
            last_selected_fp = list(fp_tanimoto_sim.keys())[id_]
            filter_data.append(fp_mol[last_selected_fp])
            fp_tanimoto_sim.pop(last_selected_fp)
            fp_mol.pop(last_selected_fp)
        return filter_data

    def generate_augmented_data(self, loader: AbstractLoader):
        chunks = self._get_chunks(range(len(loader)), loader)
        with ProcessPoolExecutor(self.featurization_jobs) as executor:
            for result in executor.map(self.apply_transform_on_loader, chunks):
                self.aug_dataset += [r for res in result for r in res]
        return self

    def apply_transform_on_loader(self, loader: AbstractLoader):
        aug_data = []
        with progress_bar() as progress:
            for data in progress.track(loader):
                aug_data.append(self.apply_transform(data))
        return aug_data

    def generate_splits(self, splits):
        self.mapper_data_aug = {}
        for e in self.aug_dataset:
            list_id = self.mapper_data_aug.get(e.original_data_id, [])
            self.mapper_data_aug.update({e.original_data_id: list_id + [e.id_]})

        for k, v in splits.copy().items():
            self.splits[f"{k}_aug"] = [e for id_ in v for e in self.mapper_data_aug[id_]]

    def get_aug_split_name(self, split_name, streamer=None, subset_id=None, subset_distributions=None):
        if subset_id is not None and subset_distributions is not None:
            return self.get_name_and_generate_subset_split(split_name, streamer, subset_id, subset_distributions)

        return f"{split_name}_aug"

    def get_name_and_generate_subset_split(self, split_name, streamer, subset_id=None, subset_distributions=None):
        subset_split_name = f"{split_name}_{subset_id}_aug"
        if subset_split_name not in self.splits.keys():
            indices = streamer._get_indices(split_name, subset_id, subset_distributions)
            self.splits[subset_split_name] = [e for id_ in indices for e in self.mapper_data_aug[id_]]

        return subset_split_name

    def __call__(self, data):
        return self.apply_transform(data)

    def __str__(self):
        return "MotifRemoval(similarity_threshold = {})".format(self.similarity_threshold)


class DecoySelectionStaticAugmentation(AbstractStaticAugmentation):
    def __init__(
        self,
        decoy_to_add,
        negative_threshold=0.3,
        positive_threshold=0.5,
        data_loader_smile_field="smiles",
        data_loader_target_field="target_sequence",
        script_file_path=f"data/cache/smiles_decoy_map_{datetime.now().strftime('%x').replace('/', '_')}",
        cfg_script=None,
        *args,
        **kwargs,
    ):
        """
        In case path_to_smiles_decoy_map is given the decoy_loader will be overwrite by the smiles/decoy/tanimoto file.
        @param decoy_to_add: Define which type of decoy to add to the dataset either positive, negative or all.
        @param negative_threshold: Tanimoto similarity threshold under which a decoy is considered negative.
        @param positive_threshold: Tanimoto similarity threshold over which a decoy is considered positive.
        @param data_loader_smile_field: Name of the dataset column containing the smiles.
        @param data_loader_target_field: Name of the dataset column containing the target sequence.
        @param script_file_path: (Optional) Fill either this parameter or `cfg_script`. Path to an output of the
            decoy_similarity script.
        @param cfg_script: (Optional) Fill either this parameter or `script_file_path`.
                            Dict containing information to generate the decoy_similarity script.
        """
        super().__init__(*args, **kwargs)
        self.data_loader_smile_field = data_loader_smile_field
        self.data_loader_target_field = data_loader_target_field
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.script_file_path = script_file_path
        self.cfg_script = cfg_script
        self.decoy_to_add = decoy_to_add

    def generate_augmented_data(self, data_loader: AbstractLoader):
        """
        We are computing the negative and positive decoys and storing them in a file
        """
        self.protein_decoy = {}
        logging.info("Computing decoys")
        smiles_decoy_map = self.get_pos_neg_decoy()
        self.protein_decoy = self.pair_decoy_target(data_loader, smiles_decoy_map)
        return self

    def get_pos_neg_decoy(self):
        """
        Run script or load the tanimoto script
        Compute the tanimoto similarity between the smiles of interest and the
        decoy candidate.
        :return: DataFrame with columns smiles, decoy_smiles, tanimoto_score
        """
        # Use a list of smiles / decoy element.
        if os.path.exists(self.script_file_path):
            smiles_tanimoto_df = self.get_decoys_from_file()
        elif self.cfg_script is not None:
            smiles_tanimoto_df = SuperFactory.create(AbstractScript, self.cfg_script).run()

        return smiles_tanimoto_df

    def get_decoys_from_file(self):
        smiles_decoy_tanimoto_df = pd.read_csv(self.script_file_path)
        pos_decoy = smiles_decoy_tanimoto_df[smiles_decoy_tanimoto_df.tanimoto_score > self.positive_threshold]
        pos_decoy = pos_decoy.groupby("smiles")["decoy_smiles"].apply(list).rename("positive_decoy")
        neg_decoy = smiles_decoy_tanimoto_df[smiles_decoy_tanimoto_df.tanimoto_score < self.negative_threshold]
        neg_decoy = neg_decoy.groupby("smiles")["decoy_smiles"].apply(list).rename("negative_decoy")
        return pd.concat([pos_decoy, neg_decoy], axis=1)  # .to_dict('index')

    def pair_decoy_target(self, data: AbstractLoader, smiles_decoy_map):
        """
        Create a dict of dict containing

        create a new loader CSV with the created df dataset
        End
        """

        filter_smile = data._dataset[np.isin(data._dataset.smiles, smiles_decoy_map.index)]
        filter_smile = filter_smile.rename(columns={"index": "original_id"})
        decoy_df = filter_smile.set_index(self.data_loader_smile_field).join(smiles_decoy_map)

        pos_decoy = self.process_decoy(data, decoy_df, "positive_decoy")
        neg_decoy = self.process_decoy(data, decoy_df, "negative_decoy")

        if self.decoy_to_add == "all":
            decoys = pd.concat([pos_decoy, neg_decoy])
            # Remove duplicates between pos and neg
            decoys = decoys.drop_duplicates(subset=[self.data_loader_target_field, self.data_loader_smile_field], keep=False)
        elif self.decoy_to_add == "positive":
            decoys = pos_decoy
            # TODO drop ones present in negative
        elif self.decoy_to_add == "negative":
            decoys = neg_decoy
            # TODO drop ones present in positive
        else:
            raise ValueError(f"self.decoy_to_add should be in ['all', 'positive', 'negative'], got '{self.decoy_to_add}'")
        decoys = decoys.reset_index(drop=True)
        self.aug_dataset = deepcopy(data)
        self._aug_dataset_df: pd.DataFrame = decoys
        self.aug_dataset._dataset = decoys
        return self.aug_dataset

    def process_decoy(self, data, decoy_df, name_column):
        subset_decoy = decoy_df.loc[
            :, [self.data_loader_target_field, name_column, "original_id"] + data._target_columns
        ].dropna()
        if len(data._target_columns) > 1:
            subset_decoy = subset_decoy[subset_decoy.loc[:, data._target_columns].astype(bool).any(axis=1)]
        else:
            subset_decoy = subset_decoy[subset_decoy.loc[:, data._target_columns].astype(bool).values]
        subset_decoy = subset_decoy.explode(name_column)
        subset_decoy = subset_decoy.drop_duplicates(subset=[self.data_loader_target_field, name_column])
        subset_decoy = subset_decoy.reset_index()
        return subset_decoy.rename(
            columns={self.data_loader_smile_field: "original_smiles", name_column: self.data_loader_smile_field}
        )

    def generate_splits(self, splits):
        self.mapper_data_aug = {}
        # Get id if other static augmentation
        start_id = self.aug_dataset[0].id_

        self.mapper_data_aug = (
            self._aug_dataset_df.reset_index()
            .groupby("original_id")
            .apply(lambda x: x.loc[:, "index"].values + start_id)
            .to_dict()
        )

        for k, v in splits.copy().items():
            self.splits[f"{k}_aug"] = [e for id_ in v for e in self.mapper_data_aug.get(id_, [])]

    def get_aug_split_name(self, split_name, streamer=None, subset_id=None, subset_distributions=None):
        if subset_id is not None and subset_distributions is not None:
            return self.get_name_and_generate_subset_split(split_name, streamer, subset_id, subset_distributions)

        return f"{split_name}_aug"

    def get_name_and_generate_subset_split(self, split_name, streamer, subset_id=None, subset_distributions=None):
        subset_split_name = f"{split_name}_{subset_id}_aug"
        if subset_split_name not in self.splits.keys():
            indices = streamer._get_indices(split_name, subset_id, subset_distributions)
            self.splits[subset_split_name] = [e for id_ in indices for e in self.mapper_data_aug[id_]]

        return subset_split_name
