from typing import Any, Dict, List
import torch
from torch import nn
from . import EnsembleNetwork, AbstractNetwork


class PseudoLroddNetwork(EnsembleNetwork):
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
            "likelihood_ratio": fg_output/bg_output,
        }


class GenerativeLSTM(AbstractNetwork):
    def __init__(self, in_features, hidden_features, out_features, num_layers=1):
        super(GenerativeLSTM, self).__init__()
        self.hidden_size = hidden_features
        self.num_layers = num_layers
        self.embedding = nn.Embedding(in_features, hidden_features)
        self.lstm = nn.LSTM(hidden_features, hidden_features, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)

        return {
            "logits": output,
            "ll": -torch.nn.functional.cross_entropy(output[:, :-1, :], x),
        }

class LroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        classifier_output, fg_output, bg_output = [model.forward(data) for model in self.models]

        return {
            "logits": classifier_output,
            "lr": fg_output["ll"] - bg_output["ll"],
        }


