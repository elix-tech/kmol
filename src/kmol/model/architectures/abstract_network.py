from abc import ABCMeta, abstractmethod

from typing import Dict, Any, Optional, List

import torch

from ...core.helpers import Namespace
from ...core.logger import LOGGER as logging
from ...core.observers import EventManager
from ...core.exceptions import CheckpointNotFound


class AbstractNetwork(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get_requirements(self) -> List[str]:
        raise NotImplementedError

    def map(self, module: "AbstractNetwork", *args) -> Dict[str, Any]:
        requirements = module.get_requirements()

        if len(args) != len(requirements):
            raise AttributeError("Cannot map inputs to module")

        return {requirement: args[index] for index, requirement in enumerate(requirements)}

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if checkpoint_path is None:
            raise CheckpointNotFound

        if device is None:
            device = torch.device("cpu")

        logging.info("Restoring from Checkpoint: {}".format(checkpoint_path))
        info = torch.load(checkpoint_path, map_location=device)

        payload = Namespace(network=self, info=info)
        EventManager.dispatch_event(event_name="before_model_checkpoint_load", payload=payload)

        self.load_state_dict(info["model"], strict=False)

    @staticmethod
    def dropout_layer_switch(m, dropout_prob):
        if isinstance(m, torch.nn.Dropout):
            if dropout_prob is not None:
                m.p = dropout_prob
            m.train()

    def activate_dropout(self, dropout_prob):
        self.apply(lambda m: self.dropout_layer_switch(m, dropout_prob))

    def mc_dropout(self, data, dropout_prob=None, n_iter=5, loss_type=""):
        self.activate_dropout(dropout_prob)

        outputs = torch.stack([self.forward(data) for _ in range(n_iter)], dim=0)

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            outputs = torch.sigmoid(outputs)

        return {"logits": torch.mean(outputs, dim=0), "logits_var": torch.var(outputs, dim=0)}

    def evidential_classification_multilabel_logits(self, data):
        outputs = self.forward(data)
        out = torch.sigmoid(outputs)
        out = torch.unsqueeze(out, dim=-1)
        out = torch.cat((out, 1 - out), -1)
        alpha = out + 1
        uncertainty = 2 / torch.sum(alpha, dim=-1, keepdim=True)
        return {"logits": outputs, "logits_var": uncertainty}

    @torch.no_grad()
    def evidential_nologits_outputs_processing(self, outputs):
        true_logits, false_logits = torch.chunk(outputs, 2, dim=-1)
        true_logits = torch.unsqueeze(true_logits, dim=-1)
        false_logits = torch.unsqueeze(false_logits, dim=-1)
        out = torch.cat((true_logits, false_logits), dim=-1)

        return torch.argmin(out, dim=-1)

    @torch.no_grad()
    def evidential_regression_outputs_processing(self, outputs):
        mu, v, alpha, beta = torch.chunk(outputs, 4, dim=-1)
        return mu

    @torch.no_grad()
    def pass_outputs(self, outputs):
        return outputs

    @torch.no_grad()
    def simple_classification_outputs_processing(self, outputs):
        return torch.argmax(outputs, dim=-1)

    def evidential_classification_multilabel_nologits(self, data):
        outputs = self.forward(data)

        true_logits, false_logits = torch.chunk(outputs, 2, dim=-1)
        true_logits = torch.unsqueeze(true_logits, dim=-1)
        false_logits = torch.unsqueeze(false_logits, dim=-1)
        out = torch.cat((true_logits, false_logits), dim=-1)

        evidence = torch.nn.functional.relu(out)
        alpha = evidence + 1
        uncertainty = (2 / torch.sum(alpha, dim=-1, keepdim=True)).squeeze()

        # logic is reversed as 0 is true and 1 is false
        prediction = torch.argmin(out, dim=-1)
        softmax_out = torch.softmax(out, dim=-1)
        softmax_score, max_indice = torch.max(softmax_out, dim=-1)

        return {"logits": prediction, "logits_var": uncertainty, "softmax_score": softmax_score}

    def evidential_classification(self, data):
        outputs = self.forward(data)
        evidence = torch.nn.functional.relu(outputs)

        alpha = evidence + 1
        uncertainty = outputs.size()[-1] / torch.sum(alpha, dim=-1, keepdim=True)
        uncertainty = uncertainty.unsqueeze(-1).repeat(1, 1, outputs.size(-1))

        return {"logits": outputs, "logits_var": uncertainty}

    def evidential_regression(self, data):
        outputs = self.forward(data)
        mu, v, alpha, beta = torch.chunk(outputs, 4, dim=-1)

        v = torch.abs(v) + 1.0
        alpha = torch.abs(alpha) + 1.0
        beta = torch.abs(beta) + 0.1

        epistemic = beta / (v * (alpha - 1))

        return {"logits": mu, "logits_var": epistemic}
