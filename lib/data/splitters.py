import math
import random
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import pandas as pd

from lib.core.exceptions import SplitError
from lib.data.loaders import AbstractLoader


class AbstractSplitter(metaclass=ABCMeta):
    """Splitters take a loader as input and return lists of entry IDs"""

    def __init__(self, splits: Dict[str, float]):
        self.splits = splits

        if round(sum(self.splits.values()), 2) != 1:
            SplitError("Split ratios do not add up to 1!")

    @abstractmethod
    def apply(self, data_loader: AbstractLoader) -> Dict[str, List[Union[int, str]]]:
        raise NotImplementedError


class IndexSplitter(AbstractSplitter):
    """Split the dataset based on their order"""

    def apply(self, data_loader: AbstractLoader) -> Dict[str, List[Union[int, str]]]:
        splits = {}

        ids = data_loader.list_ids()
        dataset_size = len(ids)

        start_index = 0
        total_ratio = 0
        for split_name, split_ratio in self.splits.items():
            total_ratio += split_ratio
            end_index = math.floor(dataset_size * total_ratio)

            splits[split_name] = ids[start_index:end_index]
            start_index = end_index

        return splits


class RandomSplitter(AbstractSplitter):
    """Split the dataset randomly"""

    def __init__(self, splits: Dict[str, float], seed: int):
        super().__init__(splits=splits)
        random.seed(seed)

    def apply(self, data_loader: AbstractLoader) -> Dict[str, List[Union[int, str]]]:
        splits = {}
        ids = set(data_loader.list_ids())

        ratio_left = 1
        for split_name, split_ratio in self.splits.items():
            sample_size = int(split_ratio / ratio_left * len(ids))
            splits[split_name] = random.sample(population=ids, k=sample_size)

            ids = ids.difference(set(splits[split_name]))
            ratio_left -= split_ratio

        return splits


class StratifiedSplitter(AbstractSplitter):
    """
    Preserve the proportion of samples based on a certain target/label.
    If the target is continuous, we can split it into a number of bins.
    """

    def __init__(self, splits: Dict[str, float], seed: int, target_name: str, bins_count: int = 0):
        super().__init__(splits=splits)

        self._seed = seed
        self._target_name = target_name
        self._bins_count = bins_count

    def _load_labels(self, data_loader: AbstractLoader) -> Dict[Union[int, str], float]:
        labels = {}

        for entry in iter(data_loader):
            labels[entry.id_] = entry.outputs[self._target_name]

        return labels

    def _binify(self, data: Dict[Union[int, str], float]) -> Dict[Union[int, str], int]:
        entries = list(data.values())
        bins = pd.cut(entries, self._bins_count, labels=False).to_list()

        return dict(zip(list(data.keys()), bins))

    def apply(self, data_loader: AbstractLoader) -> Dict[str, List[Union[int, str]]]:
        from sklearn.model_selection import train_test_split

        leftover_data = self._load_labels(data_loader)
        if self._bins_count > 0:
            leftover_data = self._binify(leftover_data)

        splits = {}
        ratio_left = 1

        for split_name, split_ratio in self.splits.items():
            current_ratio = split_ratio / ratio_left

            if current_ratio < 1:
                current_split_ids, leftover_ids = train_test_split(
                    list(leftover_data.keys()), train_size=current_ratio,
                    random_state=self._seed, stratify=list(leftover_data.values())
                )
            else:
                current_split_ids = list(leftover_data.keys())
                leftover_ids = []

            splits[split_name] = current_split_ids
            leftover_data = {id_: leftover_data[id_] for id_ in leftover_ids}

            ratio_left -= split_ratio

        return splits
