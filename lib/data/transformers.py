from abc import ABCMeta, abstractmethod
from typing import List
import logging
from lib.core.exceptions import TransformerError
import numpy as np

from lib.data.resources import DataPoint


class AbstractTransformer(metaclass=ABCMeta):

    @abstractmethod
    def apply(self, data: DataPoint) -> None:
        raise NotImplementedError

    @abstractmethod
    def reverse(self, data: DataPoint) -> None:
        raise NotImplementedError


class LogNormalizeTransformer(AbstractTransformer):

    def __init__(self, targets: List[int]):
        self._targets = targets
        self._epsilon = 1e-8

    def apply(self, data: DataPoint) -> None:
        for target in self._targets:
            data.outputs[target] = np.log(data.outputs[target] + self._epsilon)

    def reverse(self, data: DataPoint) -> None:
        for target in self._targets:
            data.outputs[target] = np.exp(data.outputs[target])


class MinMaxNormalizeTransformer(AbstractTransformer):

    def __init__(self, target: int, minimum: float, maximum: float):
        self._target = target
        self._minimum = minimum
        self._maximum = maximum

    def apply(self, data: DataPoint) -> None:
        data.outputs[self._target] = (data.outputs[self._target] - self._minimum) / (self._maximum - self._minimum)

    def reverse(self, data: DataPoint) -> None:
        data.outputs[self._target] = data.outputs[self._target] * (self._maximum - self._minimum) + self._minimum


class FixedNormalizeTransformer(AbstractTransformer):

    def __init__(self, targets: List[int], value: float):
        self._targets = targets
        self._value = value

    def apply(self, data: DataPoint) -> None:
        for target in self._targets:
            data.outputs[target] = round(data.outputs[target] / self._value, 8)

    def reverse(self, data: DataPoint) -> None:
        for target in self._targets:
            data.outputs[target] = round(data.outputs[target] * self._value, 8)


class StandardizeTransformer(AbstractTransformer):

    def __init__(self, target: int, mean: float, std: float):
        self._target = target
        self._mean = mean
        self._std = std

    def apply(self, data: DataPoint) -> None:
        data.outputs[self._target] = (data.outputs[self._target] - self._mean) / self._std

    def reverse(self, data: DataPoint) -> None:
        data.outputs[self._target] = data.outputs[self._target] * self._std + self._mean


class CutoffTransformer(AbstractTransformer):

    def __init__(self, target: int, cutoff: float):
        self._target = target
        self._cutoff = cutoff

        logging.warning("[WARNING] The cutoff transformer is destructive and cannot be reversed.")

    def apply(self, data: DataPoint) -> None:
        data.outputs[self._target] = np.where(data.outputs[self._target] < self._cutoff, 0, 1)

    def reverse(self, data: DataPoint) -> None:
        pass


class OneHotTransformer(AbstractTransformer):

    def __init__(self, target: int, classes: List[str]):
        self._target = target
        self._classes = classes

    def apply(self, data: DataPoint) -> None:
        try:
            data.outputs[self._target] = self._classes.index(data.outputs[self._target])
        except ValueError:
            raise TransformerError(
                "One-Hot Transformer Failed: '{}' is not a valid class".format(data.outputs[self._target])
            )

    def reverse(self, data: DataPoint) -> None:
        data.outputs[self._target] = self._classes[data.outputs[self._target]]
