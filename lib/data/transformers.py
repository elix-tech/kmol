from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from lib.data.resources import Data


class AbstractTransformer(metaclass=ABCMeta):

    @abstractmethod
    def apply(self, data: Data) -> None:
        raise NotImplementedError

    @abstractmethod
    def reverse(self, data: Data) -> None:
        raise NotImplementedError


class LogNormalizeTransformer(AbstractTransformer):

    def __init__(self, targets: List[int]):
        self._targets = targets
        self._epsilon = 1e-8

    def apply(self, data: Data) -> None:
        for target in self._targets:
            data.outputs[target] = np.log(data.outputs[target] + self._epsilon)

    def reverse(self, data: Data) -> None:
        for target in self._targets:
            data.outputs[target] = np.exp(data.outputs[target])


class MinMaxNormalizeTransformer(AbstractTransformer):

    def __init__(self, target: int, minimum: float, maximum: float):
        self._target = target
        self._minimum = minimum
        self._maximum = maximum

    def apply(self, data: Data) -> None:
        data.outputs[self._target] = (data.outputs[self._target] - self._minimum) / (self._maximum - self._minimum)

    def reverse(self, data: Data) -> None:
        data.outputs[self._target] = data.outputs[self._target] * (self._maximum - self._minimum) + self._minimum


class StandardizeTransformer(AbstractTransformer):

    def __init__(self, target: int, mean: float, std: float):
        self._target = target
        self._mean = mean
        self._std = std

    def apply(self, data: Data) -> None:
        data.outputs[self._target] = (data.outputs[self._target] - self._mean) / self._std

    def reverse(self, data: Data) -> None:
        data.outputs[self._target] = data.outputs[self._target] * self._std + self._mean
