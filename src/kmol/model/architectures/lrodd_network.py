from typing import Any, Dict, List
import torch
from torch import nn
from kmol.model.architectures import EnsembleNetwork
from kmol.model.architectures.abstract_network import AbstractNetwork


class PseudoLroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        """
        This is not a direct implementation for Likelihood Ratios for Out-of-Distribution Detection.
        This gives pseudo likelihood ratios between two trained classifier.
        The ratio of classifer probabilities can be interpreted as a likelihood ratio.

        Config example:
        "model": {
            "type": "pseudo_lrodd",
            "model_configs": [
                {
                    "type": "classifier",
                    "in_features": 1024,
                    "hidden_features": 512,
                    "out_features": 2,
                },
                {
                    "type": "classifier",
                    "in_features": 1024,
                    "hidden_features": 512,
                    "out_features": 2,
                }
            ]
        },
        """
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any], loss_type: str = None) -> Dict[str, torch.Tensor]:
        fg_output, bg_output = [model.forward(data) for model in self.models]

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            fg_output = torch.sigmoid(fg_output)
            bg_output = torch.sigmoid(bg_output)

        return {
            "logits": fg_output,
            "likelihood_ratio": fg_output / bg_output,
        }


class GenerativeLstmNetwork(AbstractNetwork):
    def __init__(
        self,
        in_features: int,
        n_embedding: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(in_features, n_embedding)
        self.lstm = nn.LSTM(n_embedding, hidden_features, 1, batch_first=True)
        self.fc = nn.Linear(hidden_features, out_features)

    def get_requirements(self) -> List[str]:
        return ["protein_index", "mask"]

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = data[self.get_requirements()[0]]

        x = torch.where(x == -1, torch.zeros_like(x), x)
        x = self.embedding(x)

        output, _ = self.lstm(x)
        output = self.fc(output)

        return output

    def log_forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        forward including log likelihood
        """
        output = self.forward(data)

        input = data[self.get_requirements()[0]]
        mask = input != -1
        input[~mask] = 0

        loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), input.view(-1), reduction="none")

        loss = loss.view_as(input) * mask

        log_likelihood = -loss.sum(dim=1) / mask.sum(dim=1)

        return {
            "logits": output,
            "log_likelihood": log_likelihood,
        }


class LroddNetwork(EnsembleNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        """
        Architecture used LRODD for Likelihood Ratios for Out-of-Distribution Detection (https://arxiv.org/abs/1906.02845).
        Expects a list of 3 models: classifier, foreground, background.
        Foreground and background models should be generative models.
        Background model should be trained with pertubation.
        The log likelihood difference between the foreground and background model is used to compute the likelihood ratio.
        This likelihood ratio is a metric akin to uncertainty.

        Config example:
        "model": {
            "type": "lrodd",
            "model_configs": [
                {
                    "type": "classifier",
                    "in_features": 1024,
                    "hidden_features": 512,
                    "out_features": 2,
                },
                {
                    "type": "generative_lstm",
                    "in_features": 21,
                    "n_embedding": 64,
                    "hidden_features": 64,
                    "out_features": 21
                },
                {
                    "type": "generative_lstm",
                    "in_features": 21,
                    "n_embedding": 64,
                    "hidden_features": 64,
                    "out_features": 21
                }
            ]
        },
        """
        super().__init__(model_configs)

    def forward(self, data: Dict[str, Any], loss_type: str) -> Dict[str, torch.Tensor]:
        classifier_model, fg_model, bg_model = self.models
        classifier_output = classifier_model.forward(data)
        fg_output = fg_model.log_forward(data)
        bg_output = bg_model.log_forward(data)

        likelihood_ratio = fg_output["log_likelihood"] - bg_output["log_likelihood"]

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            classifier_output = torch.sigmoid(classifier_output)

        return {
            "logits": classifier_output,
            "likelihood_ratio": likelihood_ratio,
        }
