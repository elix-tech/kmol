from typing import List
from torch.utils.data.dataset import Dataset, Subset
from ..core.helpers import SuperFactory

from .augmentations import AbstractAugmentation
from .compositions import Compose
from .preprocessor import AbstractPreprocessor
from .resources import DataPoint


class DatasetAugment(Subset):
    def __init__(self, dataset: Dataset, indices: List[int], augmentations: List=[]) -> None:
        super().__init__(dataset, indices)
        self.has_augmentation = len(augmentations) > 0
        if self.has_augmentation:
            augmentations = [
                SuperFactory.create(AbstractAugmentation, augmentation)
                for augmentation in augmentations
            ]
            self.augmentations = Compose(augmentations)
        self.training_mode = True

    def __getitem__(self, idx: int):
        data = self.dataset[self.indices[idx]]
        return self._apply_aug(data)

    def _apply_aug(self, data: DataPoint):
        if self.training_mode and self.has_augmentation:
            data.inputs = self.augmentations(data.inputs)
        return data

    def set_training_mode(self, training_mode: bool):
        self.training_mode = training_mode


class DatasetOnline(DatasetAugment):
    def __init__(self, dataset: Dataset, indices: List[int], preprocessor: AbstractPreprocessor, augmentations: List=[]) -> None:
        super().__init__(dataset, indices, augmentations)
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int):
        data = self.dataset[self.indices[idx]]
        data = self.preprocessor.preprocess(data)
        return self._apply_aug(data)