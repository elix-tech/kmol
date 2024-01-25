from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import numpy as np


from kmol.core.exceptions import TransformerError
from kmol.core.logger import LOGGER as logging
from kmol.data.resources import DataPoint


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


class AutoEncoderTransformer(AbstractTransformer):
    def __init__(self, input: str):
        self._input = input

    def apply(self, data: DataPoint) -> None:
        data.outputs = data.inputs[self._input]

    def reverse(self, data: DataPoint) -> None:
        pass


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


class MultitaskCutoffTransformer(AbstractTransformer):
    def __init__(self, cutoff: float):
        self._cutoff = cutoff

        logging.warning("[WARNING] The cutoff transformer is destructive and cannot be reversed.")

    def apply(self, data: DataPoint) -> None:
        data.outputs = list(map(self.apply_itemwise, data.outputs))

    def reverse(self, data: DataPoint) -> None:
        pass

    def apply_itemwise(self, x):
        if np.isnan(x):
            return x
        else:
            return np.ndarray.item(np.where(x < self._cutoff, 0, 1))


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


class TernaryTransformer(AbstractTransformer):
    """
    From raw continuous values, creates ternary classification labels:
    - 0 if value < mean - width (negative)
    - 1 if mean - width <= value <= mean + width (intermediate)
    - 2 if value > mean + width (positive)
    """

    def __init__(self, target: int, mean: float, width: float):
        self._target = target
        self._mean = mean
        self._width = width
        self._labels = [0, 1, 2]

    def apply(self, data: DataPoint) -> None:
        data.outputs[self._target] = np.select(self.get_conditions(data.outputs[self._target]), self._labels)

    def reverse(self, data: DataPoint) -> None:
        pass

    def get_conditions(self, query: float) -> List[Tuple]:
        return [
            (query < (self._mean - self._width)),
            (query >= (self._mean - self._width)) & (query <= (self._mean + self._width)),
            (query > self._mean + self._width),
        ]


class MultitaskTernaryTransformer(AbstractTransformer):
    """
    From raw continuous values, creates ternary classification labels:
    - 0 if value < mean - width (negative)
    - 1 if mean - width <= value <= mean + width (intermediate)
    - 2 if value > mean + width (positive)
    """

    def __init__(self, mean: float, width: float):
        self._mean = mean
        self._width = width
        self._labels = [0, 1, 2]

    def apply(self, data: DataPoint) -> None:
        data.outputs = list(map(self.apply_itemwise, data.outputs))

    def reverse(self, data: DataPoint) -> None:
        pass

    def apply_itemwise(self, x):
        if np.isnan(x):
            return x
        else:
            return np.ndarray.item(np.select(self.get_conditions(x), self._labels))

    def get_conditions(self, query: float) -> List[Tuple]:
        return [
            (query < (self._mean - self._width)),
            (query >= (self._mean - self._width)) & (query <= (self._mean + self._width)),
            (query > self._mean + self._width),
        ]
