from typing import Any, Dict, List
import torch
from torch import nn
from . import EnsembleNetwork
from .abstract_network import AbstractNetwork


class PseudoLroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        fg_output, bg_output = [model.forward(data) for model in self.models]

        return {
            "logits": fg_output,
            "likelihood_ratio": fg_output/bg_output,
        }

    def loss_aware_forward(self, data: Dict[str, Any], loss_type: str) -> Dict[str, torch.Tensor]:
        fg_output, bg_output = [model.forward(data) for model in self.models]

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            fg_output = torch.sigmoid(fg_output)
            bg_output = torch.sigmoid(bg_output)

        return {
            "logits": fg_output,
            "likelihood_ratio": fg_output/bg_output,
        }


class GenerativeLstmNetwork(AbstractNetwork):
    def __init__(
        self, 
        in_features: int,
        hidden_features: int,
        out_features: int, 
    ):
        super().__init__()
        self.hidden_size = hidden_features
        self.embedding = nn.Embedding(in_features, hidden_features)
        self.lstm = nn.LSTM(hidden_features, hidden_features, 1, batch_first=True)
        self.fc = nn.Linear(hidden_features, out_features)

    def get_requirements(self) -> List[str]:
        return ["protein", "mask"]

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        input = data["protein"]
        mask = data["mask"]

        x = self.embedding(input)

        output, _ = self.lstm(x) #, hidden)
        output = self.fc(output)

        # Use the mask to ignore padded elements when calculating the loss
        masked_output = torch.masked_select(output[:, :-1, :], mask)
        masked_input = torch.masked_select(input, mask)

        log_likelihood = -torch.nn.functional.cross_entropy(masked_output, masked_input)

        return {
            "logits": output,
            "log_likelihood": log_likelihood,
        }

class LroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        classifier_output, fg_output, bg_output = [model.forward(data) for model in self.models]

        return {
            "logits": classifier_output,
            "likelihood_ratio": fg_output["log_likelihood"] - bg_output["log_likelihood"],
        }


