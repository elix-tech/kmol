from abc import ABCMeta, abstractmethod
from typing import Iterator, Iterable, Literal, Any, List

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as ClassicDataLoader
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as GeometricDataLoader, Data as GeometricData
from torch_geometric.datasets import MoleculeNet

from lib.config import Config
from lib.helpers import Tokenizer
from lib.resources import ProteinLigandBatch


class AbstractLoader(Iterable, metaclass=ABCMeta):

    def __init__(self, config: Config, mode: Literal["train", "test"]):
        self._config = config
        self._mode = mode

        self._dataset = self._load()

    @abstractmethod
    def _load(self) -> Dataset:
        raise NotImplementedError

    def get_size(self) -> int:
        return len(self._dataset)

    def _get_indices(self, entry_count: int) -> List[int]:
        train_set_size = round(self._config.train_ratio * entry_count)

        if self._config.split_method == "index":
            indices = range(train_set_size) if self._mode == "train" else range(train_set_size, entry_count)
            indices = list(indices)
        elif self._config.split_method == "random":
            train_indices, test_indices = train_test_split(
                range(entry_count), train_size=train_set_size, random_state=self._config.seed
            )

            indices = train_indices if self._mode == "train" else test_indices
        else:
            raise ValueError("Split method not implemented for this loader: {}".format(self._config.split_method))

        if self._mode == "train":
            if sum(self._config.subset_distributions) != 1:
                raise ValueError("Subset distributions don't sum up to 1")

            remaining_entries_count = len(indices)
            subset_distribution = self._config.subset_distributions
            start_index = int(remaining_entries_count * sum(subset_distribution[:self._config.subset_id]))
            end_index = int(remaining_entries_count * sum(subset_distribution[:self._config.subset_id + 1]))

            indices = indices[start_index:end_index]

        return indices


class MoleculeNetLoader(AbstractLoader):

    def _load(self) -> MoleculeNet:
        dataset = MoleculeNet(root=self._config.input_path, name=self._config.dataset)
        indices = self._get_indices(dataset.len())

        return dataset[indices]

    def __iter__(self) -> Iterator[GeometricData]:
        data_loader = GeometricDataLoader(
            self._dataset,
            batch_size=self._config.batch_size,
            shuffle=self._mode == "train"
        )

        return iter(data_loader)


class ProteinLigandDataset(Dataset):

    def __init__(self, data: pd.DataFrame, config: Config):
        self._dataset = data
        self._config = config

        unique_proteins = self._dataset["protein_id"].unique()
        self._protein_tokenizer = Tokenizer(vocabulary=unique_proteins)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple:
        sample = self._dataset.iloc[index]
        label = [sample["label"]]

        ligand_features = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sample["smiles"]), 2)
        protein_features = self._protein_tokenizer.tokenize([sample["protein_id"]])

        return label, ligand_features, protein_features


class ProteinLigandLoader(AbstractLoader):

    def _load(self) -> ProteinLigandDataset:
        dataset = pd.read_csv(self._config.input_path)
        indices = self._get_indices(len(dataset))

        dataset = dataset.iloc[indices]
        return ProteinLigandDataset(data=dataset, config=self._config)

    def _collate_function(self, data: List[List[Any]]) -> ProteinLigandBatch:
        return ProteinLigandBatch(
            labels=torch.Tensor([sample[0] for sample in data]),
            ligand_features=torch.Tensor([sample[1] for sample in data]),
            protein_features=torch.Tensor([sample[2] for sample in data])
        )

    def __iter__(self) -> Iterator[ProteinLigandBatch]:
        data_loader = ClassicDataLoader(
            self._dataset,
            batch_size=self._config.batch_size,
            shuffle=self._mode == "train",
            collate_fn=self._collate_function
        )

        return iter(data_loader)
