import datetime
import hashlib
import json
import logging
import os
import timeit
from typing import Type, Any, Dict, Union, T

import humps
import torch

from lib.core.exceptions import ReflectionError


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
    def get_descendants(parent: Type[T]) -> Dict[str, Type[T]]:
        descendants = {}

        for subclass in parent.__subclasses__():
            if "Abstract" not in subclass.__name__:
                descendants[subclass.__name__] = subclass
            else:
                descendants = {**descendants, **SuperFactory.get_descendants(subclass)}

        return descendants

    @staticmethod
    def create(required_type: Type[Any], option_values: Dict[str, Any]) -> Any:
        """
        The super factory is a mix between an abstract factory and a dependency injector.
        If an abstract class is requested, we check all child classes which extend that abstract class
        and select the one specified by the "type" option. We then proceed to set all other attributes
        based on the "option_values". If one of the attributes is expected to be another object, it will
        be created in a recursive manner. The typing of the constructor is important for dependency injection to work!
        """

        logging.debug("Super Factory --- Required Type --- {}".format(required_type))
        logging.debug("Super Factory --- Option Values --- {}".format(option_values))
        logging.debug("------------------------------------------------------------")

        option_values = option_values.copy()
        if "type" in option_values:

            if type(required_type) is type(Union) and type(None) in required_type.__args__:
                required_type = required_type.__args__[0]  # Fix for Optional arguments

            option_key = required_type.__name__.replace("Abstract", "")

            dependency_name = "{}_{}".format(option_values["type"], option_key)
            dependency_name = humps.pascalize(dependency_name)
            option_values.pop("type")

            subclasses = SuperFactory.get_descendants(required_type)
            if dependency_name not in subclasses:
                raise ReflectionError("Dependency not found: {}. Available options are: {}".format(
                    dependency_name, subclasses.keys())
                )

            required_type = subclasses.get(dependency_name)

        if len(option_values) > 0:
            attributes = required_type.__init__.__annotations__
            for option_name, option_value in option_values.items():
                if option_name not in attributes:
                    raise ReflectionError("Unknown option for [{}] ---> [{}]".format(required_type.__name__, option_name))

                if (
                        type(option_value) is dict  # dictionaries usually mean additional objects
                        and not (  # except when the input is expected to actually be a dictionary
                            hasattr(attributes[option_name], "_name") and
                            attributes[option_name]._name == "Dict"
                        )
                ):
                    option_values[option_name] = SuperFactory.create(attributes[option_name], option_value)

        return required_type(**option_values)

    @staticmethod
    def select(required_dependency: str, available_options: Dict[str, object]) -> Any:
        if required_dependency not in available_options:
            raise ReflectionError("Dependency not found: {}. Available options are: {}".format(
                required_dependency, available_options.keys())
            )

        return available_options.get(required_dependency)


class CacheManager:

    def __init__(self, cache_location: str):
        self._cache_location = cache_location

        if not os.path.exists(self._cache_location):
            os.makedirs(self._cache_location)

    def _sort(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in dictionary.items():
            if type(value) is dict:
                dictionary[key] = self._sort(value)

        return dict(sorted(dictionary.items()))

    def key(self, **kwargs) -> str:
        options = self._sort(kwargs)
        options = json.dumps(options)

        return hashlib.md5(options.encode("utf-8")).hexdigest()

    def has(self, key: str) -> bool:
        return os.path.isfile("{}/{}".format(self._cache_location, key))

    def load(self, key: str) -> Any:
        return torch.load("{}/{}".format(self._cache_location, key))

    def save(self, data: Any, key: str) -> Any:
        return torch.save(data, "{}/{}".format(self._cache_location, key))

    def delete(self, key: str) -> None:
        try:
            os.remove("{}/{}".format(self._cache_location, key))
        except FileNotFoundError:
            pass
