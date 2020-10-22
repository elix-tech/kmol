from abc import ABCMeta, abstractmethod
from typing import Iterator, Iterable, Literal, Any

from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet

from lib.config import Config


class AbstractLoader(Iterable, metaclass=ABCMeta):

    def __init__(self, config: Config, mode: Literal["train", "test"]):
        self._config = config
        self._mode = mode

    @abstractmethod
    def _get_split(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_feature_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_class_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError


class MoleculeNetLoader(AbstractLoader):

    def __init__(self, config: Config, mode: Literal["train", "test"]):
        super().__init__(config, mode)

        dataset = MoleculeNet(root=self._config.input_path, name=self._config.dataset)
        self._dataset = self._get_split(dataset)

    def _get_split(self, dataset: MoleculeNet) -> MoleculeNet:
        entry_count = dataset.len()
        train_set_size = round(self._config.train_ratio * entry_count)

        if self._config.split_method == "index":
            dataset = dataset[:train_set_size] if self._mode == "train" else dataset[train_set_size:]
        elif self._config.split_method == "random":
            train_indices, test_indices = train_test_split(
                range(entry_count), train_size=train_set_size, random_state=self._config.seed
            )

            dataset = dataset[train_indices] if self._mode == "train" else dataset[test_indices]
        else:
            raise ValueError("Split method not implemented for this loader: {}".format(self._config.split_method))

        return dataset

    def get_feature_count(self) -> int:
        return self._dataset.num_node_features

    def get_class_count(self) -> int:
        return self._dataset.num_classes

    def get_size(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator:
        data_loader = DataLoader(self._dataset, batch_size=self._config.batch_size, shuffle=self._mode == "train")
        return iter(data_loader)
