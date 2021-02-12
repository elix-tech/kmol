import logging
import operator
from abc import ABCMeta, abstractmethod
from copy import copy
from enum import Enum
from functools import reduce
from typing import List, Dict, Union

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lib.core.config import Config
from lib.core.exceptions import FeaturizationError
from lib.core.helpers import SuperFactory, CacheManager
from lib.data.featurizers import AbstractFeaturizer
from lib.data.loaders import AbstractLoader, ListLoader
from lib.data.resources import Data, Collater
from lib.data.splitters import AbstractSplitter
from lib.data.transformers import AbstractTransformer


class AbstractStreamer(metaclass=ABCMeta):

    def __init__(self):
        self._dataset = self._load_dataset()

    @property
    def labels(self) -> List[str]:
        return self._dataset.get_labels()

    @abstractmethod
    def _load_dataset(self) -> AbstractLoader:
        raise NotImplementedError

    @abstractmethod
    def get(self, split_name: str, shuffle: bool, batch_size: int) -> DataLoader:
        raise NotImplementedError


class GeneralStreamer(AbstractStreamer):

    def __init__(self, config: Config):
        self._config = config
        self._cache_manager = CacheManager(cache_location=self._config.cache_location)

        self._featurizers = [
            SuperFactory.create(AbstractFeaturizer, featurizer) for featurizer in self._config.featurizers
        ]

        self._transformers = [
            SuperFactory.create(AbstractTransformer, transformer) for transformer in self._config.transformers
        ]

        self._dataset = self._load_dataset()
        self._splits = self._generate_splits()

    def _generate_splits(self) -> Dict[str, List[Union[int, str]]]:
        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter)
        return splitter.apply(data_loader=self._dataset)

    def _load_dataset(self) -> AbstractLoader:
        cache_key = self._cache_manager.key(
            loader=self._config.loader, featurizers=self._config.featurizers, transformers=self._config.transformers
        )

        if self._config.clear_cache:
            self._cache_manager.delete(cache_key)

        if self._cache_manager.has(cache_key):
            return self._cache_manager.load(cache_key)

        dataset = self._prepare_dataset()
        self._cache_manager.save(dataset, cache_key)

        return dataset

    def _featurize(self, sample: Data):
        for featurizer in self._featurizers:
            try:
                featurizer.run(sample)
            except (FeaturizationError, ValueError, IndexError, AttributeError) as e:
                raise FeaturizationError("[WARNING] Could not run featurizer '{}' on '{}' --- {}".format(
                    featurizer.__class__.__name__, sample.id_, e
                ))

    def _apply_transformers(self, sample: Data) -> None:
        for transformer in self._transformers:
            transformer.apply(sample)

    def reverse_transformers(self, sample: Data) -> None:
        for transformer in reversed(self._transformers):
            transformer.reverse(sample)

    def _prepare_dataset(self) -> ListLoader:

        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        logging.info("Starting featurization...")

        dataset = []
        ids = []

        with tqdm(total=len(loader)) as progress_bar:
            for sample in loader:
                try:
                    self._featurize(sample)
                    self._apply_transformers(sample)

                    dataset.append(sample)
                    ids.append(sample.id_)

                except FeaturizationError as e:
                    logging.warning(e)

                progress_bar.update(1)

        dataset = ListLoader(dataset, ids)
        return dataset

    def _get_subset(self, split_name: str, **kwargs) -> Subset:
        return Subset(dataset=self._dataset, indices=self._splits[split_name])

    def get(self, split_name: str, batch_size: int, shuffle: bool, **kwargs) -> DataLoader:
        collater = Collater(device=self._config.get_device())

        return DataLoader(
            dataset=self._get_subset(split_name, **kwargs),
            collate_fn=collater.apply,
            batch_size=batch_size,
            shuffle=shuffle
        )


class SubsetStreamer(GeneralStreamer):

    def _get_subset(self, split_name: str, subset_id: int, subset_distributions: List[float]) -> Subset:
        indices = self._splits[split_name]

        remaining_entries_count = len(indices)
        start_index = int(remaining_entries_count * sum(subset_distributions[:subset_id]))
        end_index = int(remaining_entries_count * sum(subset_distributions[:subset_id + 1]))

        return Subset(dataset=self._dataset, indices=indices[start_index:end_index])


class CrossValidationStreamer(GeneralStreamer):

    class Mode(Enum):
        TRAIN = "train"
        TEST = "test"

    def get_fold_name(self, fold: int) -> str:
        return "fold_{}".format(fold)

    def _generate_splits(self) -> Dict[str, List[str]]:
        split_ratio = 1 / self._config.cross_validation_folds
        splits = {self.get_fold_name(fold): split_ratio for fold in range(self._config.cross_validation_folds)}

        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter, {"splits": splits})
        return splitter.apply(data_loader=self._dataset)

    def _get_subset(self, split_name: str, mode: Mode) -> Subset:
        if mode == self.Mode.TEST:
            indices = self._splits[split_name]
        else:
            indices = copy(self._splits)
            indices.pop(split_name)
            indices = reduce(operator.iconcat, indices.values(), [])  # flatten

        return Subset(dataset=self._dataset, indices=indices)
