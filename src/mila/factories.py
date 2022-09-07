from copy import deepcopy
import json
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class AbstractConfiguration(metaclass=ABCMeta):

    output_path: str

    @classmethod
    def from_json(cls, file_path: str) -> "AbstractConfiguration":
        # with open(file_path) as read_handle:
        #     return cls(**json.load(read_handle))
        def load_from_other_config(file_path, path_to_load: List[str]):
            dir_path = Path(file_path).parent
            base_cfg = {}
            for path in path_to_load:
                path = dir_path.joinpath(path).resolve()
                tmp_cfg = get_dict_from_json(path)
                base_cfg = cls.update_recursive_dict(base_cfg, tmp_cfg)
            return base_cfg

        def get_dict_from_json(file_path):
            with open(file_path) as read_handle:
                cfg = json.load(read_handle)
            if '__load__' in cfg.keys():
                base_cfg = load_from_other_config(file_path, cfg['__load__'])
                cfg = cls.update_recursive_dict(base_cfg, cfg)
                del cfg['__load__']
            return cfg

        cfg = get_dict_from_json(file_path)
        return cls(**cfg)

    @classmethod
    def update_recursive_dict(cls, current, update):
        output = deepcopy(current)
        for k, v in update.items():
            if type(v) == dict:
                if type(current.get(k, None)) == dict:
                    output.update({
                        k: cls.update_recursive_dict(current.get(k, {}), v)
                    })
                else: # is None
                    output.update({k: v})
            else:
                output.update({k: v})

        return output


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
