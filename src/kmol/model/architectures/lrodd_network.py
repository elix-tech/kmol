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
        return ["protein_index", "mask"]

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = data[self.get_requirements()[0]]
        
        x = self.embedding(x)

        output, _ = self.lstm(x) #, hidden)
        output = self.fc(output)

        return output

    def lr_forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        output = self.forward(data)

        input = data[self.get_requirements()[0]]
        mask = data[self.get_requirements()[1]]

        loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), input.view(-1), reduction="none")
        
        loss = loss.view_as(input) * mask

        log_likelihood = -loss.sum() / mask.sum()

        return {
            "logits": output,
            "log_likelihood": log_likelihood,
        }
        

class LroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        classifier_model, fg_model, bg_model = self.models
        classifier_output = classifier_model.forward(data)
        fg_output = fg_model.lr_forward(data)
        bg_output = bg_model.lr_forward(data)
        
        return {
            "logits": classifier_output,
            "likelihood_ratio": fg_output["log_likelihood"] - bg_output["log_likelihood"],
        }


