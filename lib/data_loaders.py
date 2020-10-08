from abc import ABCMeta, abstractmethod
from typing import Iterable, Iterator, NamedTuple, List

import numpy as np
import pandas as pd
from rdkit import Chem

from lib.config import Config
from lib.featurizers import AtomFeaturizer


class DataPoint(NamedTuple):
    node_features: np.ndarray
    adjacency_matrix: np.ndarray
    labels: np.ndarray


class AbstractLoader(Iterable, metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def get_feature_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_class_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError

    def _load_batch(self, buffer: Iterator[DataPoint]) -> List[DataPoint]:
        batch = []

        while len(batch) < self._config.batch_size:
            try:
                batch.append(next(buffer))
            except StopIteration:
                break

        return batch


class CsvLoader(AbstractLoader):

    def __init__(self, config: Config):
        super().__init__(config)
        self._dataset = self._featurize(config.input_path)

    def _featurize(self, file_path: str) -> List[DataPoint]:
        dataset = pd.read_csv(file_path)

        featurizer = AtomFeaturizer()
        data_points = []

        for entry in dataset.itertuples():
            smiles = getattr(entry, self._config.input_field)
            mol = Chem.MolFromSmiles(smiles)

            labels = [getattr(entry, field) for field in self._config.target_fields]
            labels = [float(label) if label else float("Nan") for label in labels]

            data_points.append(DataPoint(
                node_features=featurizer.apply(mol),
                adjacency_matrix=Chem.GetAdjacencyMatrix(mol),
                labels=np.array(labels)
            ))

        return data_points

    def _get_labels(self, entry: NamedTuple) -> List[float]:
        labels = [getattr(entry, field) for field in self._config.target_fields]
        labels = [float(label) if label else float("Nan") for label in labels]

        return labels

    def get_feature_count(self) -> int:
        return self._dataset[0].node_features.shape[0]

    def get_class_count(self) -> int:
        return len(self._config.target_fields)

    def get_size(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator:

