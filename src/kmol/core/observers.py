from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List

import torch
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormLayer

from .helpers import Namespace


class AbstractEventHandler(metaclass=ABCMeta):

    @abstractmethod
    def run(self, payload: Namespace):
        raise NotImplementedError


class EventManager:
    _LISTENERS: DefaultDict[str, List[AbstractEventHandler]] = defaultdict(list)

    @staticmethod
    def add_event_listener(event_name: str, handler: AbstractEventHandler) -> None:
        EventManager._LISTENERS[event_name].append(handler)

    @staticmethod
    def dispatch_event(event_name: str, payload: Namespace) -> None:
        for handler in EventManager._LISTENERS[event_name]:
            handler.run(payload=payload)

    @staticmethod
    def flush() -> None:
        EventManager._LISTENERS = defaultdict(list)


class AddSigmoidEventHandler(AbstractEventHandler):
    """event: before_criterion|before_predict"""

    def run(self, payload: Namespace):
        payload.logits = torch.sigmoid(payload.logits)


class AddSoftmaxEventHandler(AbstractEventHandler):
    """event: before_criterion|before_predict"""

    def run(self, payload: Namespace):
        payload.logits = torch.softmax(payload.logits, dim=-1)


class InjectLossWeightsEventHandler(AbstractEventHandler):
    """event: before_criterion"""

    def __init__(self, mappers: List[str]):
        self._mappers = mappers

    def run(self, payload: Namespace):
        for mapper in self._mappers:
            payload.extras.append(payload.features.inputs[mapper].reshape(-1, 1))


class DropParametersEventHandler(AbstractEventHandler):
    """event: before_checkpoint_load"""

    def __init__(self, keywords: List[str]):
        self._keywords = keywords

    def run(self, payload: Namespace):
        for keyword in self._keywords:
            try:
                del payload.info["model"][keyword]
            except KeyError:
                pass


class DropBatchNormLayersEventHandler(AbstractEventHandler):
    """event: various"""

    def run(self, payload: Namespace) -> None:
        from opacus.utils.module_modification import nullify_batchnorm_modules
        nullify_batchnorm_modules(payload.executor.network)


class ReplaceBatchNormLayersEventHandler(AbstractEventHandler):
    """event: various"""

    def converter(self, module: BatchNormLayer) -> torch.nn.Module:
        return torch.nn.GroupNorm(module.num_features, module.num_features, affine=True)

    def run(self, payload: Namespace) -> None:
        from opacus.utils.module_modification import replace_all_modules
        replace_all_modules(payload.executor.network, BatchNormLayer, self.converter)


class DifferentialPrivacy:

    class AttachPrivacyEngineEventHandler(AbstractEventHandler):
        """event: before_train_start"""

        def __init__(self, **kwargs):
            self._options = kwargs

            if "alphas" not in self._options:
                self._options["alphas"] = [1 + i / 10.0 for i in range(1, 100)] + list(range(12, 64))

        def run(self, payload: Namespace) -> None:
            from vendor.opacus.custom.privacy_engine import PrivacyEngine

            trainer = payload.trainer
            network = trainer.network

            if not isinstance(self._options["max_grad_norm"], list):
                self._options["max_grad_norm"] = [self._options["max_grad_norm"]] * len(list(network.parameters()))

            privacy_engine = PrivacyEngine(
                module=network,
                batch_size=trainer.config.batch_size,
                sample_size=len(payload.data_loader.dataset),
                **self._options
            )

            privacy_engine.attach(trainer.optimizer)

    class LogPrivacyCostEventHandler(AbstractEventHandler):
        """event: before_train_progress_log"""

        def __init__(self, delta: float):
            self._delta = delta

        def run(self, payload: Namespace) -> None:
            optimizer = payload.trainer.optimizer

            try:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self._delta)
                payload.message += " - privacy_cost: (ε = {:.2f}, δ = {}, α = {})".format(
                    epsilon, self._delta, best_alpha
                )
            except AttributeError:
                pass

    @staticmethod
    def setup(delta: float = 1e-5, **kwargs):
        EventManager.add_event_listener(
            "before_train_start", DifferentialPrivacy.AttachPrivacyEngineEventHandler(**kwargs)
        )

        EventManager.add_event_listener(
            "before_train_progress_log", DifferentialPrivacy.LogPrivacyCostEventHandler(delta)
        )
