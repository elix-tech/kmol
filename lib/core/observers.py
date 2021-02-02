from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List, Type

from lib.core.helpers import Namespace


class EventHandler(metaclass=ABCMeta):

    @abstractmethod
    def run(self, payload: Namespace):
        raise NotImplementedError


class EventManager:
    _LISTENERS: DefaultDict[str, List[Type[EventHandler]]] = defaultdict(list)

    @staticmethod
    def add_event_listener(event_name: str, handler: Type[EventHandler]) -> None:
        EventManager._LISTENERS[event_name].append(handler)

    @staticmethod
    def dispatch_event(event_name: str, payload: Namespace) -> None:
        for handler in EventManager._LISTENERS[event_name]:
            payload = handler().run(payload=payload)


class MaskMissingLabelsHandler(EventHandler):

    def run(self, payload: Namespace):
        mask = payload.target == payload.target
        weights = mask.float()
        labels = payload.target
        labels[~mask] = 0

        payload.target = labels
        payload.weight = weights
