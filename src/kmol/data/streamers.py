from copy import copy
import operator
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial, reduce
from typing import List, Dict, Union

from torch.utils.data import DataLoader, Subset

from .preprocessor import AbstractPreprocessor, OnlinePreprocessor
from .datasets import DatasetAugment, DatasetOnline
from .resources import DataPoint, AbstractCollater, LoadedContent
from .splitters import AbstractSplitter
from ..core.config import Config
from ..core.helpers import SuperFactory


class AbstractStreamer(metaclass=ABCMeta):
    @property
    def labels(self) -> List[str]:
        return self._dataset.get_labels()

    @abstractmethod
    def get(self, split_name: str, shuffle: bool, batch_size: int) -> DataLoader:
        raise NotImplementedError


class GeneralStreamer(AbstractStreamer):
    class Mode(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self, config: Config):
        self._config = config
        self._preprocessor: AbstractPreprocessor = SuperFactory.create(
            AbstractPreprocessor, self._config.preprocessor, loaded_parameters={"config": self._config}
        )

        self._collater = SuperFactory.create(AbstractCollater, self._config.collater)

        self._dataset = self._preprocessor._load_dataset()
        self._preprocessor._load_augmented_data()
        self.splits = self._generate_splits()
        self._generate_aug_splits()
        self._dataset_object = self._generate_partial_dataset_object()

    def _generate_splits(self) -> Dict[str, List[Union[int, str]]]:
        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter)
        return splitter.apply(data_loader=self._dataset)

    def _generate_aug_splits(self) -> Dict[str, List[Union[int, str]]]:
        for a in self._preprocessor._static_augmentations:
            a.generate_splits(self.splits)

    def reverse_transformers(self, sample: DataPoint) -> None:
        return self._preprocessor.reverse_transformers(sample)

    def _get_subset(self, split_name: str, mode: Mode, **kwargs) -> Subset:
        # Retrieve base data and indices
        indices = self._get_indices(split_name, mode, **kwargs)
        # Retrieve static aug data and indices
        if mode == self.Mode.TRAIN:
            aug_indices = self._get_aug_indices(split_name, **kwargs)
            indices += aug_indices
            dataset = self._preprocessor._add_static_aug_dataset(self._dataset)
        else:
            dataset = self._dataset

        dataset = self._dataset_object(dataset=dataset, indices=indices)
        dataset.set_training_mode(mode == self.Mode.TRAIN)
        return dataset

    def _get_indices(self, split_name, mode, **kwargs):
        return self.splits[split_name]

    def _get_aug_indices(self, split_name, **kwargs):
        splits = []
        for a in self._preprocessor._static_augmentations:
            splits += a.splits[a.get_aug_split_name(split_name)]
        return splits

    def _generate_partial_dataset_object(self) -> Subset:
        if isinstance(self._preprocessor, OnlinePreprocessor):
            return partial(DatasetOnline, augmentations=self._config.augmentations, preprocessor=self._preprocessor)
        else:
            return partial(DatasetAugment, augmentations=self._config.augmentations)

    def get(self, split_name: str, batch_size: int, shuffle: bool, mode: Mode, **kwargs) -> LoadedContent:
        data_loader = DataLoader(
            dataset=self._get_subset(split_name, mode, **kwargs),
            collate_fn=self._collater.apply,
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
    def _get_indices(
        self, split_name: str, mode: GeneralStreamer.Mode, subset_id: int, subset_distributions: List[float]
    ) -> List:
        indices = self.splits[split_name]

        remaining_entries_count = len(indices)
        start_index = int(remaining_entries_count * sum(subset_distributions[:subset_id]))
        end_index = int(remaining_entries_count * sum(subset_distributions[: subset_id + 1]))

        return indices[start_index:end_index]

    def _get_aug_indices(self, split_name: str, subset_id: int, subset_distributions: List[float]) -> List:
        splits = []
        for a in self._preprocessor._static_augmentations:
            splits += a.splits[a.get_aug_split_name(split_name, self, subset_id, subset_distributions)]
        return splits


class CrossValidationStreamer(GeneralStreamer):
    def get_fold_name(self, fold: int) -> str:
        return "fold_{}".format(fold)

    def _generate_splits(self) -> Dict[str, List[str]]:
        split_ratio = 1 / self._config.cross_validation_folds
        splits = {self.get_fold_name(fold): split_ratio for fold in range(self._config.cross_validation_folds)}

        splitter = SuperFactory.create(AbstractSplitter, self._config.splitter, {"splits": splits})
        return splitter.apply(data_loader=self._dataset)

    def _get_indices(self, split_name: str, mode: GeneralStreamer.Mode) -> List:
        if mode == self.Mode.TEST:
            indices = self.splits[split_name]
        else:
            indices = copy(self.splits)
            indices.pop(split_name)
            indices = reduce(operator.iconcat, indices.values(), [])  # flatten

        return indices

    def _get_aug_indices(self, split_name: str) -> List:
        aug_indices = []
        for a in self._preprocessor._static_augmentations:
            aug_split = copy(a.splits)
            aug_split.pop(split_name + "_aug")
            aug_indices += reduce(operator.iconcat, aug_split.values(), [])  # flatten
        return aug_indices
