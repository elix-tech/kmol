import json
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class AbstractConfiguration(metaclass=ABCMeta):

    output_path: str

    @classmethod
    def from_json(cls, file_path: str) -> "AbstractConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))


class AbstractExecutor(metaclass=ABCMeta):

    def __init__(self, config: AbstractConfiguration):
        self._config = config

    def run(self, job: str):
        if not hasattr(self, job):
            raise ValueError("Unknown job requested: {}".format(job))

        getattr(self, job)()

    @abstractmethod
    def train(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> Any:
        raise NotImplementedError


class AbstractAggregator(metaclass=ABCMeta):

    @abstractmethod
    def run(self, checkpoint_paths: List[str], save_path: str) -> None:
        raise NotImplementedError
