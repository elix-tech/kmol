import logging
from abc import ABCMeta, abstractmethod
from typing import Union, List

from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

from lib.core.config import Config
from lib.core.exceptions import FeaturizationError
from lib.core.helpers import SuperFactory, CacheManager
from lib.data.featurizers import AbstractFeaturizer
from lib.data.loaders import AbstractLoader, ListLoader
from lib.data.resources import Data, Collater
from lib.data.splitters import AbstractSplitter


class AbstractStreamer(metaclass=ABCMeta):

    @abstractmethod
    def get(self, shuffle: bool, batch_size: int) -> DataLoader:
        raise NotImplementedError


class GeneralStreamer(AbstractStreamer):

    def __init__(self, config: Config, split_name: str):
        self._config = config
        self._split_name = split_name

        self._cache_manager = CacheManager(cache_location=self._config.cache_location)
        self._dataset = self._load_dataset()

    def _load_dataset(self) -> AbstractLoader:
        cache_key = self._cache_manager.key(
            loader=self._config.loader, splitter=self._config.splitter,
            featurizers=self._config.featurizers, split_name=self._split_name
        )

        if self._config.clear_cache:
            self._cache_manager.delete(cache_key)

        if self._cache_manager.has(cache_key):
            return self._cache_manager.load(cache_key)

        dataset = self._prepare_dataset()
        self._cache_manager.save(dataset, cache_key)

        return dataset

    def _featurize(self, sample: Data, featurizers: List[AbstractFeaturizer]) -> Data:
        for featurizer in featurizers:
            try:
                sample = featurizer.run(sample)
            except (FeaturizationError, ValueError, IndexError, AttributeError) as e:
                raise FeaturizationError("[WARNING] Could not apply '{}' on '{}' --- {}".format(
                    featurizer.__class__.__name__, sample.id_, e
                ))

        return sample

    def _prepare_dataset(self) -> ListLoader:

        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter)
        featurizers = [SuperFactory.create(AbstractFeaturizer, featurizer) for featurizer in self._config.featurizers]

        splits = splitter.apply(data_loader=loader)
        sample_ids = splits[self._split_name]

        logging.info("Starting featurization...")
        dataset = []

        with tqdm(total=len(sample_ids)) as progress_bar:
            for id_ in sample_ids:

                try:
                    sample = loader[id_]
                    sample = self._featurize(sample, featurizers)
                    dataset.append(sample)

                except FeaturizationError as e:
                    logging.warning(e)

                progress_bar.update(1)

        dataset = ListLoader(dataset)
        return dataset

    def _get_dataset(self, **kwargs) -> AbstractLoader:
        return self._dataset

    def get(self, batch_size: int, shuffle: bool, **kwargs) -> DataLoader:
        collater = Collater()

        return DataLoader(
            dataset=self._get_dataset(**kwargs),
            collate_fn=collater.apply,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )


class SubsetStreamer(GeneralStreamer):

    def _get_dataset(self, subset_id: int, subset_distributions: List[float]) -> Subset:
        indices = self._dataset.list_ids()

        remaining_entries_count = len(indices)
        start_index = int(remaining_entries_count * sum(subset_distributions[:subset_id]))
        end_index = int(remaining_entries_count * sum(subset_distributions[:subset_id + 1]))

        return Subset(dataset=self._dataset, indices=indices[start_index:end_index])


class CrossValidationStreamer(SubsetStreamer):

    def _get_dataset(self, k: int, subset_id: int) -> Subset:
        subset_distributions = [1 / k] * k
        return super()._get_dataset(subset_id=subset_id, subset_distributions=subset_distributions)


class LeaveOneOutCrossValidationStreamer(CrossValidationStreamer):

    def _get_dataset(self, subset_id: int) -> Subset:
        return super()._get_dataset(subset_id=subset_id, k=len(self._dataset))
