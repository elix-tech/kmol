import logging
import operator
import os
from abc import ABCMeta, abstractmethod
from copy import copy
from enum import Enum
from functools import reduce
from joblib import Parallel, delayed
from typing import List, Dict, Union

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .featurizers import AbstractFeaturizer
from .loaders import AbstractLoader, ListLoader
from .resources import DataPoint, Collater, LoadedContent
from .splitters import AbstractSplitter
from .transformers import AbstractTransformer
from ..core.config import Config
from ..core.exceptions import FeaturizationError
from ..core.helpers import SuperFactory, CacheManager


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
            SuperFactory.create(AbstractFeaturizer, featurizer)
            for featurizer in self._config.featurizers
        ]

        self._transformers = [
            SuperFactory.create(AbstractTransformer, transformer)
            for transformer in self._config.transformers
        ]

        self._dataset = self._load_dataset()
        self.splits = self._generate_splits()

    def _generate_splits(self) -> Dict[str, List[Union[int, str]]]:
        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter)
        return splitter.apply(data_loader=self._dataset)

    def _load_dataset(self) -> AbstractLoader:
        return self._cache_manager.execute_cached_operation(
            processor=self._prepare_dataset,
            clear_cache=self._config.clear_cache,
            arguments={},
            cache_key={
                "loader": self._config.loader,
                "featurizers": self._config.featurizers,
                "transformers": self._config.transformers,
                "last_modified": os.path.getmtime(self._config.loader["input_path"])
            }
        )

    def _featurize(self, sample: DataPoint):
        for featurizer in self._featurizers:
            try:
                featurizer.run(sample)
            except (
                FeaturizationError,
                ValueError,
                IndexError,
                AttributeError,
                TypeError,
            ) as e:
                raise FeaturizationError(
                    "[WARNING] Could not run featurizer '{}' on '{}' --- {}".format(
                        featurizer.__class__.__name__, sample.id_, e
                    )
                )

    def _apply_transformers(self, sample: DataPoint) -> None:
        for transformer in self._transformers:
            transformer.apply(sample)

    def reverse_transformers(self, sample: DataPoint) -> None:
        for transformer in reversed(self._transformers):
            transformer.reverse(sample)

    def _prepare_dataset(self) -> ListLoader:

        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        all_ids = loader.list_ids()
        n_jobs = self._config.featurization_jobs
        chunk_size = len(all_ids) // n_jobs
        logging.info("Starting featurization...")
        chunks = [
            all_ids[i:i + chunk_size] for i in range(0, len(all_ids), chunk_size)
        ]
        chunks = [Subset(loader, chunk) for chunk in chunks]
        dataset = sum(
            Parallel(n_jobs=len(chunks))(
                delayed(self._prepare_chunk)(chunk) for chunk in chunks
            ),
            [],
        )

        ids = [sample.id_ for sample in dataset]
        return ListLoader(dataset, ids)

    def _prepare_chunk(self, loader) -> List[DataPoint]:
        dataset = []
        with tqdm(total=len(loader)) as progress_bar:
            for sample in loader:
                try:
                    self._featurize(sample)
                    self._apply_transformers(sample)

                    dataset.append(sample)

                except FeaturizationError as e:
                    logging.warning(e)

                progress_bar.update(1)

        return dataset

    def _get_subset(self, split_name: str, **kwargs) -> Subset:
        return Subset(dataset=self._dataset, indices=self.splits[split_name])

    def get(
        self, split_name: str, batch_size: int, shuffle: bool, **kwargs
    ) -> LoadedContent:
        collater = Collater()

        data_loader = DataLoader(
            dataset=self._get_subset(split_name, **kwargs),
            collate_fn=collater.apply,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._config.num_workers,
            drop_last=self._config.drop_last_batch,
            pin_memory=True,
        )

        return LoadedContent(
            dataset=data_loader,
            batches=len(data_loader),
            samples=len(data_loader.dataset),
        )


class SubsetStreamer(GeneralStreamer):
    def _get_subset(
        self, split_name: str, subset_id: int, subset_distributions: List[float]
    ) -> Subset:
        indices = self.splits[split_name]

        remaining_entries_count = len(indices)
        start_index = int(
            remaining_entries_count * sum(subset_distributions[:subset_id])
        )
        end_index = int(
            remaining_entries_count * sum(subset_distributions[: subset_id + 1])
        )

        return Subset(dataset=self._dataset, indices=indices[start_index:end_index])


class CrossValidationStreamer(GeneralStreamer):
    class Mode(Enum):
        TRAIN = "train"
        TEST = "test"

    def get_fold_name(self, fold: int) -> str:
        return "fold_{}".format(fold)

    def _generate_splits(self) -> Dict[str, List[str]]:
        split_ratio = 1 / self._config.cross_validation_folds
        splits = {
            self.get_fold_name(fold): split_ratio
            for fold in range(self._config.cross_validation_folds)
        }

        splitter = SuperFactory.create(
            AbstractSplitter, self._config.splitter, {"splits": splits}
        )
        return splitter.apply(data_loader=self._dataset)

    def _get_subset(self, split_name: str, mode: Mode) -> Subset:
        if mode == self.Mode.TEST:
            indices = self.splits[split_name]
        else:
            indices = copy(self.splits)
            indices.pop(split_name)
            indices = reduce(operator.iconcat, indices.values(), [])  # flatten

        return Subset(dataset=self._dataset, indices=indices)
