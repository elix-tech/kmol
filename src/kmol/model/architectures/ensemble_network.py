from typing import Dict, Any, Optional, List

import torch

from ...core.helpers import SuperFactory
from .abstract_network import AbstractNetwork
from ...core.observers import EventManager, AddLossInfoHandler


class EnsembleNetwork(AbstractNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__()
        self.models = torch.nn.ModuleList([SuperFactory.create(AbstractNetwork, config) for config in model_configs])
        EventManager.add_event_listener(event_name="before_predict", handler=AddLossInfoHandler())

    def load_checkpoint(self, checkpoint_paths: List[str], device: Optional[torch.device] = None):
        n_models = len(self.models)
        n_checkpoints = len(checkpoint_paths)

        if n_models != n_checkpoints:
            raise ValueError(
                f"Number of checkpoint_path should be equal to number of models. Received {n_models}, {n_checkpoints}."
            )
        for model, checkpoint_path in zip(self.models, checkpoint_paths):
            model.load_checkpoint(checkpoint_path, device)

    def get_requirements(self):
        return list(set(sum([model.get_requirements() for model in self.models], [])))

    def forward(self, data: Dict[str, Any], loss_type: str = None) -> Dict[str, torch.Tensor]:
        outs = [model.forward(data) for model in self.models]
        outputs = torch.stack(outs, dim=0)

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            outputs = torch.sigmoid(outputs)

        return {
            "logits": torch.mean(outputs, dim=0),
            "logits_var": torch.var(outputs, dim=0),
        }

    def mc_dropout(
        self,
        data,
        dropout_prob=None,
        n_iter=5,
        return_distrib=False,
        loss_type="",
    ):
        self.activate_dropout(dropout_prob)

        means, vars = zip(*[model.mc_dropout(data, dropout_prob, n_iter, loss_type).values() for model in self.models])
        means = torch.stack(means, dim=0)
        mean = means.mean(dim=0)
        var = (torch.stack(vars, dim=0).mean(dim=0) + means.var(dim=0)) / 2

        return {"logits": mean, "logits_var": var}
