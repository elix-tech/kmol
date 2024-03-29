import os
import datetime
import hashlib
import importlib
import io
import json
import pickle
import os
import tempfile
import timeit
from dataclasses import dataclass
from functools import partial, total_ordering
from typing import Type, Any, Dict, Union, T, Optional, List, Callable, TextIO, BinaryIO

import humps
import numpy as np
import torch
from filelock import FileLock
from disklist import DiskList
from multiprocessing import Manager

from kmol.core.exceptions import ReflectionError
from kmol.core.logger import LOGGER as logging


class Timer:
    def __init__(self):
        self.start_time = None
        self.reset()

    def __call__(self) -> float:
        return timeit.default_timer() - self.start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), 0)))

    def reset(self) -> None:
        self.start_time = timeit.default_timer()


class SuperFactory:
    @staticmethod
    def find_descendants(parent: Type[T]) -> Dict[str, Type[T]]:
        descendants = {}

        for child in parent.__subclasses__():
            descendants[child.__name__] = child

            if len(child.__subclasses__()) > 0:
                descendants.update(SuperFactory.find_descendants(child))

        return descendants

    @staticmethod
    def create(
        instantiator: Optional[Type[Any]],
        dynamic_parameters: Dict[str, Any],
        loaded_parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        The super factory is a mix between an abstract factory and a dependency injector.
        If an abstract class is requested, we check all child classes which extend that abstract class
        and select the one specified by the "type" option. We then proceed to set all other attributes
        based on the "option_values". If one of the attributes is expected to be another object, it will
        be created in a recursive manner. The typing of the constructor is important for dependency injection to work!

        :type instantiator: The desired class, or its abstraction. Can be "None" when using reflections (see below).
        :type loaded_parameters: Additional options which are already loaded. They just get appended to the dynamic ones
        :type dynamic_parameters: A list of options which will be injected recursively. Can include a "type" option.
            if "type" is of format "kmol.package.subpackage.ObjectName" it will be reflected directly.
            if "type" is a string in "snake_case" format, a matching descendant from "instantiator" will be searched for
            ie: requesting a "very_smart" object descendant of "AbstractCalculator" will fetch the "VerySmartCalculator"
        """

        logging.debug("Super Factory --- Instantiator --- {}".format(instantiator))
        logging.debug("Super Factory --- Dynamic Parameters --- {}".format(dynamic_parameters))
        logging.debug("Super Factory --- Loaded Parameters --- {}".format(loaded_parameters))
        logging.debug("------------------------------------------------------------")

        if loaded_parameters is None:
            loaded_parameters = {}

        dynamic_parameters = dynamic_parameters.copy()
        if "type" in dynamic_parameters:
            dependency_type = dynamic_parameters.pop("type")

            if "." in dependency_type:
                instantiator = SuperFactory.reflect(dependency_type)
            else:
                if type(instantiator) is type(Union) and type(None) in instantiator.__args__:
                    # Fix for Optional arguments
                    instantiator = instantiator.__args__[0]

                option_key = instantiator.__name__.replace("Abstract", "")
                dependency_name = "{}_{}".format(dependency_type, option_key)
                dependency_name = humps.pascalize(dependency_name)

                subclasses = SuperFactory.find_descendants(instantiator)
                if dependency_name not in subclasses:
                    raise ReflectionError(
                        "Dependency not found: {}. Available options are: {}".format(dependency_name, subclasses.keys())
                    )
                instantiator = subclasses.get(dependency_name)

        if not dynamic_parameters and not loaded_parameters:
            return instantiator()

        try:
            parameters = instantiator.__init__.__code__.co_varnames
            if len(dynamic_parameters) > 0:
                attributes = instantiator.__init__.__annotations__

                for option_name, option_value in dynamic_parameters.items():
                    if option_name not in parameters and "kwargs" not in parameters:
                        raise ReflectionError("Unknown option for [{}] ---> [{}]".format(instantiator.__name__, option_name))

                    if option_name not in attributes:
                        continue  # for 3rd party libraries that don't use type hints...

                    if type(option_value) is dict and not (
                        hasattr(attributes[option_name], "_name") and attributes[option_name]._name == "Dict"
                    ):
                        # if the option is a dictionary, and the argument is not expected to be one
                        # we consider it an additional object which we instantiate/inject recursively
                        dynamic_parameters[option_name] = SuperFactory.create(attributes[option_name], option_value)

            options = dynamic_parameters
            for key, value in loaded_parameters.items():
                if key in parameters:
                    options[key] = value
        except Exception as e:
            logging.debug(f"Exception found while instantiating object: {e}")
            options = dynamic_parameters

        return instantiator(**options)

    @staticmethod
    def reflect(dependency: str) -> Type[Any]:
        logging.debug("Reflecting --- {}".format(dependency))
        logging.debug("------------------------------------------------------------")

        module_name, class_name = dependency.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


class CacheManager:
    def __init__(self, cache_location: str):
        self._cache_location = cache_location

        if not os.path.exists(self._cache_location):
            os.makedirs(self._cache_location, exist_ok=True)

    def _sort(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in sorted(dictionary.items()):
            if type(value) is dict:
                dictionary[key] = self._sort(value)
            elif type(value) is list and len(value) > 0:
                if type(value[0]) is dict:
                    dictionary[key] = [self._sort(v) for v in value]
        return dict(sorted(dictionary.items()))

    def key(self, **kwargs) -> str:
        options = self._sort(kwargs)
        options = json.dumps(options)

        return hashlib.md5(options.encode("utf-8")).hexdigest()

    def has(self, key: str) -> bool:
        return os.path.isfile("{}/{}".format(self._cache_location, key))

    def load(self, key: str) -> Any:
        try:
            with open("{}/{}".format(self._cache_location, key), "rb") as file:
                bio = io.BytesIO(pickle.load(file))
            return torch.load(bio)
        except Exception:
            return torch.load("{}/{}".format(self._cache_location, key))

    def save(self, data: Any, key: str) -> Any:
        try:
            bio = io.BytesIO()
            torch.save(data, bio)
            with open("{}/{}".format(self._cache_location, key), "wb") as file:
                pickle.dump(bio.getvalue(), file)
        except Exception:
            return torch.save(data, "{}/{}".format(self._cache_location, key))

    def delete(self, key: str) -> None:
        try:
            os.remove("{}/{}".format(self._cache_location, key))
        except FileNotFoundError:
            pass

    def execute_cached_operation(
        self,
        processor: Callable,
        arguments: Dict[str, Any],
        cache_key: Dict[str, Any],
        clear_cache: bool = False,
    ) -> Any:
        if cache_key.get("loader", {}).get("input_path", False):
            cache_key["timestamp"] = os.path.getmtime(cache_key["loader"]["input_path"])

        cache_key = self.key(**cache_key)
        logging.info(f"Dataset Cache Key: {cache_key}")

        if self.has(cache_key) and not clear_cache:
            return self.load(cache_key)

        content = processor(**arguments)

        if clear_cache:
            self.delete(cache_key)

        self.save(content, cache_key)

        return content


class Namespace:
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other: "Namespace") -> bool:
        return vars(self) == vars(other)

    def __contains__(self, key: str):
        return key in self.__dict__

    def __repr__(self) -> str:
        representation = ""
        for key, value in self.__dict__.items():
            representation += "{}={}, ".format(key, value)

        return "Namespace[{}]".format(representation[:-2])

    @staticmethod
    def reduce(namespaces: List["Namespace"], operation: Callable) -> "Namespace":
        options = {}
        for key in vars(namespaces[0]).keys():
            options[key] = operation([getattr(namespace, key) for namespace in namespaces])

        return Namespace(**options)

    @staticmethod
    def max(namespaces: List["Namespace"]) -> "Namespace":
        return Namespace.reduce(namespaces, np.maximum.reduce)

    @staticmethod
    def min(namespaces: List["Namespace"]) -> "Namespace":
        return Namespace.reduce(namespaces, np.minimum.reduce)

    @staticmethod
    def mean(namespaces: List["Namespace"]) -> "Namespace":
        return Namespace.reduce(namespaces, partial(np.mean, axis=0))


@dataclass
@total_ordering
class ConfidenceInterval:
    mean: float
    deviation: float
    confidence: float

    def __str__(self) -> str:
        return "{:.4f}±{:.4f}".format(self.mean, self.deviation)

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: "ConfidenceInterval") -> bool:
        return self.mean < other.mean

    def __eq__(self, other: "ConfidenceInterval") -> bool:
        return self.mean == other.mean and self.deviation == other.deviation and self.confidence == other.confidence

    def __add__(self, other: "ConfidenceInterval") -> "ConfidenceInterval":
        return ConfidenceInterval(
            mean=self.mean + other.mean,
            deviation=self.deviation + other.deviation,
            confidence=min(self.confidence, other.confidence),
        )

    def __truediv__(self, other: int) -> "ConfidenceInterval":
        return ConfidenceInterval(mean=self.mean / other, deviation=self.deviation / other, confidence=self.confidence)

    @staticmethod
    def compute(values: Union[List[List[float]], np.ndarray], z: float = 1.96) -> List["ConfidenceInterval"]:
        values = np.array(values).transpose()

        return [
            ConfidenceInterval(mean=np.mean(value), deviation=z * np.std(value) / np.sqrt(len(value)), confidence=z)
            for value in values
        ]


class Loggable:
    def __init__(self, file_path: str):
        self._logger = self._create_logger(file_path)

    def _create_logger(self, file_path: str) -> TextIO:
        if "/" in file_path:
            os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)

        return open(file_path, "w")

    def log(self, message: str) -> None:
        self._logger.write(message)

    def close(self) -> None:
        self._logger.close()

    def __enter__(self) -> "Loggable":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.close()


class HookProbe:
    def __init__(self, model: torch.nn.Module, module_name: Optional[str] = None):
        self.module_name = module_name if module_name != "last_hidden" else getattr(model, "last_hidden_layer_name", None)

        self._cache = []
        self.hook = None

        def get_hidden_layer_output(model, input, output):
            if isinstance(output, torch.Tensor):
                self._cache.append(output.detach())

        for name, module in model.named_modules():
            if name == self.module_name:
                self.hook = module.register_forward_hook(get_hidden_layer_output)
                break

        if self.hook is None and self.module_name is not None:
            raise ValueError(
                "Error when selecting layer to probe:"
                f" could not find '{self.module_name}' in the network modules."
                " Please check the 'probe_layer' parameter in the configuration."
                " This parameter should match one of the network module name"
                " or can be set to 'last_hidden'."
            )

    def get_probe(self):
        return torch.stack(self._cache).mean(dim=0)


class CacheDiskList(DiskList):
    def __init__(
        self,
        init_temp_file: BinaryIO = None,
        cache_size: int = -1,
        tmp_dir: Union[Any, os.PathLike, None] = None,
        delete: bool = False,
    ):
        self.item_offsets = []
        self.item_sizes = []
        self.cache = []
        self.cache_size = cache_size
        self.cache_index = 0
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.delete = delete
        self.tempfile = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=delete) if not init_temp_file else init_temp_file
        self.tempfile_name = self.tempfile.name
        self.lock = FileLock(self.tempfile.name + ".lock", thread_local=False)

    def get_lock(self):
        try:
            return self.lock
        except:
            # For already cached file support should be deletable later.
            self.lock = FileLock(self.tempfile.name + ".lock", thread_local=False)
            return self.lock

    def append(self, item):
        with self.get_lock():
            super().append(item)

    def __getitem__(self, index):
        with self.get_lock():
            return super().__getitem__(index)

    def clear(self):
        self.tempfile.close()
        os.unlink(self.tempfile_name)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tempfile"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tempfile = open(self.tempfile_name, "r+b")
