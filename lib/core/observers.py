from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List

import torch
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormLayer

from lib.core.helpers import Namespace


class EventHandler(metaclass=ABCMeta):

    @abstractmethod
    def run(self, payload: Namespace):
        raise NotImplementedError


class EventManager:
    _LISTENERS: DefaultDict[str, List[EventHandler]] = defaultdict(list)

    @staticmethod
    def add_event_listener(event_name: str, handler: EventHandler) -> None:
        EventManager._LISTENERS[event_name].append(handler)

    @staticmethod
    def dispatch_event(event_name: str, payload: Namespace) -> None:
        for handler in EventManager._LISTENERS[event_name]:
            handler.run(payload=payload)

    @staticmethod
    def flush() -> None:
        EventManager._LISTENERS = defaultdict(list)


class MaskMissingLabelsHandler(EventHandler):
    """event: before_criterion"""

    def run(self, payload: Namespace):
        mask = payload.target == payload.target
        weights = mask.float()
        labels = payload.target
        labels[~mask] = 0

        payload.target = labels
        payload.weight = weights


class DropBatchNormLayersHandler(EventHandler):
    """event: various"""

    def run(self, payload: Namespace) -> None:
        from opacus.utils.module_modification import nullify_batchnorm_modules
        nullify_batchnorm_modules(payload.executor._network)


class DropParametersHandler(EventHandler):
    """event: before_checkpoint_load"""

    def __init__(self):
        self._keywords = []

    def run(self, payload: Namespace):
        for keyword in self._keywords:
            try:
                del payload.info["model"][keyword]
            except KeyError:
                pass


class DifferentialPrivacy:

    class AttachPrivacyEngineHandler(EventHandler):
        """event: before_train_start"""

        def __init__(self, **kwargs):
            self._options = kwargs

            if "alphas" not in self._options:
                self._options["alphas"] = [1 + i / 10.0 for i in range(1, 100)] + list(range(12, 64))

        def run(self, payload: Namespace) -> None:
            from vendor.opacus.custom.privacy_engine import PrivacyEngine

            trainer = payload.trainer
            network = trainer._network

            if not isinstance(self._options["max_grad_norm"], list):
                self._options["max_grad_norm"] = [self._options["max_grad_norm"]] * len(list(network.parameters()))

            privacy_engine = PrivacyEngine(
                module=network,
                batch_size=trainer._config.batch_size,
                sample_size=len(payload.data_loader.dataset),
                **self._options
            )

            privacy_engine.attach(trainer._optimizer)

    class LogPrivacyCostHandler(EventHandler):
        """event: before_train_progress_log"""

        def __init__(self, delta: float):
            self._delta = delta

        def run(self, payload: Namespace) -> None:
            optimizer = payload.trainer._optimizer

            try:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self._delta)
                payload.message += " - privacy_cost: (ε = {:.2f}, δ = {}, α = {})".format(
                    epsilon, self._delta, best_alpha
                )
            except AttributeError:
                pass

    class ReplaceBatchNormLayersHandler(EventHandler):
        """event: after_network_create"""

        def converter(self, module: BatchNormLayer) -> torch.nn.Module:
            return torch.nn.GroupNorm(module.num_features, module.num_features, affine=True)

        def run(self, payload: Namespace) -> None:
            from opacus.utils.module_modification import replace_all_modules
            replace_all_modules(payload.executor._network, BatchNormLayer, self.converter)

    @staticmethod
    def setup(delta: float = 1e-5, **kwargs):
        EventManager.add_event_listener("after_network_create", DifferentialPrivacy.ReplaceBatchNormLayersHandler())
        EventManager.add_event_listener("before_train_start", DifferentialPrivacy.AttachPrivacyEngineHandler(**kwargs))
        EventManager.add_event_listener("before_train_progress_log", DifferentialPrivacy.LogPrivacyCostHandler(delta))
