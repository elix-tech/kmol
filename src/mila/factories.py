from copy import deepcopy
import json
import yaml
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class AbstractConfiguration(metaclass=ABCMeta):
    @classmethod
    def from_file(cls, file_path: str, job_command: str) -> "AbstractConfiguration":
        """
        Only json and YAML format are supported, config are expected to
        have the correct suffix.
        If a `__load__` key exist in the configuration, the code will expect a
        list of path to other config that needs to be imported.
        - The order of the config in the list gives the priority, ie: the last
            one will overwrite the first if there are argument conflict.
        - It is possible to overwrite only one parameter of a object if either the
            `type` argument which define the object is the same or not given.
        - It is not possible to overwrite only one element of a list. The full
            list needs to be define again in that case.
        """

        def load_from_other_config(file_path, path_to_load: List[str]):
            dir_path = Path(file_path).parent
            base_cfg = {}
            for path in path_to_load:
                if "/" not in path:  # is in the actual dir
                    path = dir_path / path
                elif path[0] == ".":  # is relative to the dir
                    path = dir_path.joinpath(path)
                tmp_cfg = get_dict_from_file(Path(path).resolve())
                base_cfg = cls._update_recursive_dict(base_cfg, tmp_cfg)
            return base_cfg

        def yaml_join(loader, node):
            """
            Join a list of string to form one argument
            """
            seq = loader.construct_sequence(node)
            return "".join([str(i) for i in seq])

        def yaml_join_path(loader, node):
            """
            Join a list of string with / to make a path
            """
            seq = loader.construct_sequence(node)
            return "/".join([str(i) for i in seq])

        def yaml_join_path_suffix(loader, node):
            """
            Join a list of string with / to make a path and use the last element
            as suffix
            """
            seq = loader.construct_sequence(node)
            seq[-2] = seq[-2] + seq[-1]
            seq.pop(-1)
            return "/".join([str(i) for i in seq])

        def get_dict_from_file(file_path):
            if Path(file_path).suffix == ".json":
                with open(file_path) as read_handle:
                    cfg = json.load(read_handle)
            elif Path(file_path).suffix in [".yaml", ".yml"]:
                yaml.add_constructor("!join", yaml_join)
                yaml.add_constructor("!path_join", yaml_join_path)
                yaml.add_constructor("!path_join_suffix", yaml_join_path_suffix)
                with open(file_path) as read_handle:
                    cfg = yaml.load(read_handle, Loader=yaml.FullLoader)
            else:
                raise ValueError("The config file should be a json or yaml with a correct suffix")
            if "__load__" in cfg.keys():
                base_cfg = load_from_other_config(file_path, cfg["__load__"])
                cfg = cls._update_recursive_dict(base_cfg, cfg)
                del cfg["__load__"]
            return cfg

        cfg = get_dict_from_file(file_path)
        if cfg.get("parameters", False):
            del cfg["parameters"]
        cfg = cls._support_old_configuration(cfg)
        return cls(job_command, **cfg)

    @classmethod
    def _update_recursive_dict(cls, current, update):
        """
        Internal method to update the configuration dict in a recursive manner.
        """
        output = deepcopy(current)
        for k, v in update.items():
            if type(v) == dict:
                # In case the type is not define or is the same the config is updated
                # otherwise it is overwritten.
                if type(current.get(k, None)) == dict and current.get("type") == v.get("type", current.get("type")):
                    output.update({k: cls._update_recursive_dict(current.get(k, {}), v)})
                else:  # is None
                    output.update({k: v})
            else:
                output.update({k: v})

        return output

    @classmethod
    def _support_old_configuration(cls, cfg):
        # Fix version 1.1.8
        if "online_preprocessing" in cfg:
            if cfg["online_preprocessing"]:
                cfg["preprocessor"] = {"type": "online"}
            else:
                cfg["preprocessor"] = {
                    "type": "cache",
                    "use_disk": cfg.get("preprocessing_use_disk", False),
                    "disk_dir": cfg.get("preprocessing_disk_dir", ""),
                }
            cfg.pop("online_preprocessing")
            cfg.pop("preprocessing_use_disk", None)
            cfg.pop("preprocessing_disk_dir", None)
        return cfg


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


class AbstractScript(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        raise NotImplementedError
