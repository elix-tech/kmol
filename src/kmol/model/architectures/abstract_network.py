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

    @torch.no_grad()
    def pass_outputs(self, outputs):
        return outputs
