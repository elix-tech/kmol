import logging
import operator
from abc import ABCMeta, abstractmethod
from copy import copy
from enum import Enum
from functools import reduce, partial
from math import ceil
from typing import List, Dict, Union, Any, Iterator

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lib.core.config import Config
from lib.core.exceptions import FeaturizationError
from lib.core.helpers import SuperFactory, CacheManager
from lib.data.featurizers import AbstractFeaturizer
from lib.data.loaders import AbstractLoader, ListLoader
from lib.data.resources import Data, Batch, Collater, LoadedContent
from lib.data.splitters import AbstractSplitter
from lib.data.transformers import AbstractTransformer


class AbstractStreamer(metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config

    @property
    def labels(self) -> List[str]:
        return self._config.loader["target_column_names"]

    @abstractmethod
    def get(self, split_name: str, shuffle: bool, batch_size: int) -> LoadedContent:
        raise NotImplementedError


class CacheIterator:

    def __init__(
            self, data_loader: DataLoader, shard_size: int, cache_manager: CacheManager,
            cache_key: Dict[str, Any], refresh_cache: bool
    ):
        self._data_loader = data_loader
        self._cache_manager = cache_manager
        self._cache_key = cache_key
        self._shard_size = shard_size
        self._refresh_cache = refresh_cache

    @property
    def shards_count(self) -> int:
        return ceil(len(self._data_loader) / self._shard_size)

    def preload(self) -> None:
        cache_key = self._cache_manager.key(**self._cache_key)
        if not self._refresh_cache and self._cache_manager.has(cache_key):
            return

        shard_id = 0
        shard = []

        logging.info("Preloading data... This can use up a lot of disk space. Make sure to monitor your cache folder.")
        for batch in tqdm(self._data_loader):
            shard.append(batch)

            if len(shard) == self._shard_size:
                shard_key = self._cache_manager.key(shard_id=shard_id, **self._cache_key)
                self._cache_manager.save(data=shard, key=shard_key)

                shard_id += 1
                shard = []

        if len(shard) > 0:
            shard_key = self._cache_manager.key(shard_id=shard_id, **self._cache_key)
            self._cache_manager.save(data=shard, key=shard_key)

        self._cache_manager.save(True, cache_key)
        self._refresh_cache = False

    def __iter__(self) -> Iterator[Batch]:
        self.preload()

        for shard_id in range(ceil(len(self._data_loader) / self._shard_size)):
            shard_key = self._cache_manager.key(shard_id=shard_id, **self._cache_key)
            shard = self._cache_manager.load(key=shard_key)

            for batch in shard:
                yield batch


class GeneralStreamer(AbstractStreamer):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._cache_manager = CacheManager(cache_location=self._config.cache_location)

        self._featurizers = [
            SuperFactory.create(AbstractFeaturizer, featurizer) for featurizer in self._config.featurizers
        ]

        self._transformers = [
            SuperFactory.create(AbstractTransformer, transformer) for transformer in self._config.transformers
        ]

        self._is_loaded = False
        self._dataset = None
        self._splits = None

    def _lazy_load(self) -> None:
        if not self._is_loaded:
            self._dataset = self._load_dataset()
            self._splits = self._generate_splits()

            self._is_loaded = True

    def _generate_splits(self) -> Dict[str, List[Union[int, str]]]:
        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter)
        return splitter.apply(data_loader=self._dataset)

    def _load_dataset(self) -> AbstractLoader:
        return self._cache_manager.execute_cached_operation(
            processor=self._prepare_dataset, clear_cache=self._config.clear_cache, arguments={}, cache_key={
                "loader": self._config.loader,
                "featurizers": self._config.featurizers,
                "transformers": self._config.transformers
            }
        )

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

    def _get_data_loader(
            self, split_name: str, batch_size: int, shuffle: bool, cache_key: Dict[str, Any], **kwargs
    ) -> LoadedContent:

        self._lazy_load()
        collater = Collater(device=self._config.get_device())

        data_loader = DataLoader(
            dataset=self._get_subset(split_name, **kwargs),
            collate_fn=collater.apply,
            batch_size=batch_size,
            shuffle=shuffle
        )

        content_instantiator = partial(LoadedContent, samples=len(data_loader.dataset), batches=len(data_loader))
        if self._config.preload_data:
            cache_key["shard_size"] = self._config.shard_size
            data_loader = CacheIterator(
                data_loader=data_loader, shard_size=32, cache_manager=self._cache_manager,
                refresh_cache=self._config.clear_cache, cache_key=cache_key
            )

            data_loader.preload()

        return content_instantiator(dataset=data_loader)

    def get(self, split_name: str, batch_size: int, shuffle: bool, **kwargs) -> LoadedContent:

        cache_key = {
            "split_name": split_name, "batch_size": batch_size, "shuffle": shuffle,
            "loader": self._config.loader, "featurizers": self._config.featurizers, "splitter": self._config.splitter,
            "transformers": self._config.transformers, **kwargs
        }

        arguments = {"split_name": split_name, "batch_size": batch_size, "shuffle": shuffle, "cache_key": cache_key}
        arguments = {**arguments, **kwargs}

        loaded_content = self._cache_manager.execute_cached_operation(
            processor=self._get_data_loader, arguments=arguments,
            clear_cache=self._config.clear_cache, cache_key=cache_key
        )

        return loaded_content


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
