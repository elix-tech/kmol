from typing import Any, Dict, List
import torch

from . import EnsembleNetwork


class LroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        fg_output, bg_output = [model.forward(data) for model in self.models]

        return {
            "logits": fg_output,
            "logits_var": fg_output - bg_output,
        }

    def loss_aware_forward(self, data: Dict[str, Any], loss_type: str) -> Dict[str, torch.Tensor]:
        fg_output, bg_output = [model.forward(data) for model in self.models]

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            fg_output = torch.sigmoid(fg_output)
            bg_output = torch.sigmoid(bg_output)

        return {
            "logits": fg_output,
            "logits_var": fg_output - bg_output,
        }
