from abc import ABCMeta, abstractmethod


class AbstractMeter(metaclass=ABCMeta):

    def __init__(self):
        self._value = 0
        self._iteration = 0

    @abstractmethod
    def update(self, value: float) -> None:
        raise NotImplementedError

    def get(self) -> float:
        return self._value

    def reset(self) -> None:
        self._iteration = 0
        self._value = 0


class AccumulationMeter(AbstractMeter):
    """Returns the sum of all the accumulated loss"""

    def update(self, value: float) -> None:
        self._iteration += 1
        self._value += value


class BatchAccumulationMeter(AccumulationMeter):
    """Also returns the sum of all the accumulated loss, but the value is reset on every read"""

    def get(self) -> float:
        value = self._value
        self.reset()

        return value


class AverageMeter(AccumulationMeter):
    """Returns the average loss per iteration/batch"""

    def get(self) -> float:
        return self._value / self._iteration


class ExponentialAverageMeter(AbstractMeter):
    """Assigns a greater weight to the most recent entries"""

    def __init__(self, smoothing_factor: float):
        super().__init__()
        self._alpha = smoothing_factor

    def update(self, value: float) -> None:
        self._iteration += 1

        if self._iteration == 1:
            self._value = value
        else:
            self._value = self._alpha * self._value + (1 - self._alpha) * value
