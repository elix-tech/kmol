from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
from typing import DefaultDict, List

import torch
from torch.nn.modules.batchnorm import _BatchNorm as BatchNormLayer

from kmol.core.helpers import Namespace
from kmol.core.logger import LOGGER as logging
from kmol.model.evidential_losses import prepare_edl_classification_output


class AbstractEventHandler(metaclass=ABCMeta):
    @abstractmethod
    def run(self, payload: Namespace):
        raise NotImplementedError


class EventManager:
    _LISTENERS: DefaultDict[str, List[AbstractEventHandler]] = defaultdict(list)

    @staticmethod
    def add_event_listener(event_name: str, handler: AbstractEventHandler, skip_if_exists: bool = False) -> None:
        if skip_if_exists:
            if any([isinstance(event, type(handler)) for event in EventManager._LISTENERS[event_name]]):
                return 
        EventManager._LISTENERS[event_name].append(handler)

    @staticmethod
    def dispatch_event(event_name: str, payload: Namespace) -> None:
        for handler in EventManager._LISTENERS[event_name]:
            handler.run(payload=payload)

    @staticmethod
    def flush() -> None:
        EventManager._LISTENERS = defaultdict(list)


class AddLogEventHandler(AbstractEventHandler):
    """event: before_metric"""

    def run(self, payload: Namespace):
        payload.logits = torch.log(payload.logits)


class AddSigmoidEventHandler(AbstractEventHandler):
    """event: before_criterion|before_predict|before_metric"""

    def run(self, payload: Namespace):
        payload.logits = torch.sigmoid(payload.logits)


class AddReluEventHandler(AbstractEventHandler):
    """event: before_criterion|before_predict"""

    def run(self, payload: Namespace):
        payload.logits = torch.nn.functional.relu(payload.logits)


class AddSoftmaxEventHandler(AbstractEventHandler):
    """event: before_criterion|before_predict"""

    def run(self, payload: Namespace):
        payload.logits = torch.softmax(payload.logits, dim=-1)


class AddArgmaxEventHandler(AbstractEventHandler):
    """event: before_predict"""

    def __init__(self, dim: int = -1, keepdim: bool = True):
        self._dim = dim
        self._keepdim = keepdim

    def run(self, payload: Namespace):
        payload.logits = torch.argmax(payload.logits, dim=self._dim, keepdim=self._keepdim)


class InjectLossWeightsEventHandler(AbstractEventHandler):
    """event: before_criterion"""

    def __init__(self, mappers: List[str]):
        self._mappers = mappers

    def run(self, payload: Namespace):
        for mapper in self._mappers:
            payload.extras.append(payload.features.inputs[mapper].reshape(-1, 1))


class AddEpochEventHandler(AbstractEventHandler):
    """event: before_criterion"""

    def run(self, payload: Namespace):
        payload.extras.append(payload.epoch)


class AddLossInfoHandler(AbstractEventHandler):
    """event: before_predict"""

    def run(self, payload: Namespace):
        payload.extras.append(payload.loss_type)


class EvidentialClassificationProcessingHandler(AbstractEventHandler):
    """event: before_tracker_update"""

    def run(self, payload: Namespace):
        outputs, alpha = prepare_edl_classification_output(payload.outputs)
        payload.outputs = torch.argmin(outputs, dim=-1)


class EvidentialRegressionProcessingHandler(AbstractEventHandler):
    """event: before_tracker_update"""

    def run(self, payload: Namespace):
        mu, v, alpha, beta = torch.chunk(payload.outputs, 4, dim=-1)
        payload.outputs = mu


class EvidentialClassificationInferenceHandler(AbstractEventHandler):
    """event: after_val_inference|after_predict"""

    def run(self, payload: Namespace):
        outputs = payload.logits
        outputs, alpha = prepare_edl_classification_output(outputs)
        uncertainty = (2 / torch.sum(alpha, dim=-1, keepdim=True)).squeeze()

        # logic is reversed as 0 is true and 1 is false
        logits = torch.log(alpha[..., 0]) - torch.log(alpha[..., 1])

        softmax_score = alpha[..., 0] / torch.sum(alpha, dim=-1)

        payload.logits = logits
        payload.logits_var = uncertainty
        payload.softmax_score = softmax_score


class EvidentialRegressionInferenceHandler(AbstractEventHandler):
    """event: after_val_inference|after_predict"""

    def run(self, payload: Namespace):
        outputs = payload.logits
        mu, v, alpha, beta = torch.chunk(outputs, 4, dim=-1)

        v = torch.abs(v) + 1.0
        alpha = torch.abs(alpha) + 1.0
        beta = torch.abs(beta) + 0.1

        epistemic = beta / (v * (alpha - 1))

        payload.logits = mu
        payload.logits_var = epistemic


class AddMaskEventHandler(AbstractEventHandler):
    """event: before_criterion"""

    def run(self, payload: Namespace):
        payload.extras.append(payload.mask)


class RemoveNansEventHandler(AbstractEventHandler):
    """event: before_metric"""

    def run(self, payload: Namespace) -> None:
        mask = payload.ground_truth == payload.ground_truth
        payload.ground_truth = torch.unsqueeze(
            torch.masked_select(payload.ground_truth, mask),
            dim=-1,
        )
        payload.logits = torch.unsqueeze(
            torch.masked_select(payload.logits, mask),
            dim=-1,
        )


class ReshapeMulticlassTensorEventHandler(AbstractEventHandler):
    """event: before_criterion"""

    def __init__(self, num_classes: int, num_tasks: int):
        self._num_classes = num_classes
        self._num_tasks = num_tasks

    def run(self, payload: Namespace) -> None:
        payload.logits = payload.logits.view(-1, self._num_classes, self._num_tasks)


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


class AddFedproxRegularizationEventHandler(AbstractEventHandler):
    """event: after_criterion"""

    def __init__(self, mu: float):
        self.mu = mu
        self.weights = None

    def get_weights(self, config) -> OrderedDict:
        if self.weights is None:
            info = torch.load(config.checkpoint_path, map_location=config.get_device())

            weights = OrderedDict()
            for key, value in info["model"].items():
                name = key.replace(".module.", ".")
                weights[name] = value

            self.weights = weights

        return self.weights

    def run(self, payload: Namespace) -> None:
        if payload.executor.config.checkpoint_path is None:
            logging.info("Skipping FedProx regularization (no checkpoint found). This is normal for the first round.")
            return

        local_weights = payload.executor.network.state_dict()
        global_weights = self.get_weights(payload.executor.config)

        regularization = 0.0
        for name, parameter in local_weights.items():
            if not name.endswith(".num_batches_tracked"):
                regularization += (self.mu / 2) * torch.norm((parameter - global_weights[name])) ** 2

        payload.loss += regularization


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
                **self._options,
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
                payload.message += " - privacy_cost: (ε = {:.2f}, δ = {}, α = {})".format(epsilon, self._delta, best_alpha)
            except AttributeError:
                pass

    @staticmethod
    def setup(delta: float = 1e-5, **kwargs):
        EventManager.add_event_listener(
            "before_train_start",
            DifferentialPrivacy.AttachPrivacyEngineEventHandler(**kwargs),
        )

        EventManager.add_event_listener(
            "before_train_progress_log",
            DifferentialPrivacy.LogPrivacyCostEventHandler(delta),
        )
