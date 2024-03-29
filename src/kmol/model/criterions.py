import torch
from typing import List, Dict

from .evidential_losses import (
    edl_classification,
    edl_classification_masked,
    edl_regression,
)

from kmol.core.observers import (
    EventManager,
    AddEpochEventHandler,
    EvidentialClassificationProcessingHandler,
    EvidentialRegressionProcessingHandler,
    EvidentialClassificationInferenceHandler,
    EvidentialRegressionInferenceHandler,
)


class WeightedLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.modules.loss._Loss):
        super().__init__()

        self._loss = loss
        self._loss.reduction = "none"

    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = self._loss(logits, ground_truth)

        loss *= weights
        loss = loss.sum(dim=0) / (weights.sum(dim=0) + 1e-8)

        loss = loss.mean()
        return loss


class MaskedLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.modules.loss._Loss):
        super().__init__()
        self._loss = WeightedLoss(loss=loss)

    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        mask = ground_truth == ground_truth
        ground_truth_copy = ground_truth.clone()  # Avoid side-effect from next line!
        ground_truth_copy[~mask] = 0
        return self._loss(logits, ground_truth_copy, mask)


class MultiTaskLoss(torch.nn.Module):
    def __init__(
        self,
        regression_loss: torch.nn.Module,
        classification_loss: torch.nn.Module,
        regression_tasks: List[int],
        classification_tasks: List[int],
        regression_weight: float = 1,
        classification_weight: float = 1,
    ):
        super().__init__()

        self._regression_loss = regression_loss
        self._classification_loss = classification_loss

        self._regression_tasks = regression_tasks
        self._classification_tasks = classification_tasks

        self._regression_weight = regression_weight
        self._classification_weight = classification_weight

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        regression_loss = self._regression_loss(outputs[:, self._regression_tasks], labels[:, self._regression_tasks])

        classification_loss = self._classification_loss(
            outputs[:, self._classification_tasks], labels[:, self._classification_tasks]
        )

        return regression_loss * self._regression_weight + classification_loss * self._classification_weight


class MultiHeadMaskedLoss(torch.nn.Module):
    def __init__(
        self,
        loss: torch.nn.Module,
        weights: List[int],
    ):
        super().__init__()
        self._loss = loss
        self._weights = weights

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, heads = outputs.shape
        if len(self._weights) != heads:
            raise ValueError("Number of weights must be equal to number of heads")

        # mask the loss
        mask = labels == labels
        labels[~mask] = 0  # set nans to 0
        mask = mask.float()

        loss = 0
        for i in range(heads):
            head_loss = torch.mul(self._loss(outputs[:, i], labels[:, i]), mask[:, i]) + 1e-8
            loss += self._weights[i] * torch.sum(head_loss) / (torch.sum(mask[:, i]) + 1e-8)
        return loss


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return super().forward(logits, ground_truth.view(-1).long())


class MultitaskCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        return super().forward(logits, ground_truth.long())


class EvidentialLoss(torch.nn.Module):
    def __init__(self, loss: Dict[str, str]):
        """
        Criterions for evidential deep learning.
        Implementation of following papers for the modes:
        - classification => Evidential Deep Learning to Quantify Classification Uncertainty  (https://arxiv.org/pdf/1806.01768.pdf)
        - regression => Deep Evidential Regression (https://arxiv.org/abs/1910.02600)

        Config example:
        "criterion": {
            "type": "kmol.model.criterions.EvidentialLoss",
            "loss": {"type": "classification", "annealing": true}
        },
        """
        super().__init__()

        self._loss_type = loss["type"]
        self._loss_annealing = loss["annealing"]

        loss_dict = {
            "classification": edl_classification,
            "classification_masked": edl_classification_masked,
            "regression": edl_regression,
        }
        self._loss = loss_dict[loss["type"]]

        EventManager.add_event_listener(event_name="before_criterion", handler=AddEpochEventHandler(), skip_if_exists=True)

        if self._loss_type == "classification" or self._loss_type == "classification_masked":
            EventManager.add_event_listener(
                event_name="before_tracker_update", handler=EvidentialClassificationProcessingHandler(), skip_if_exists=True
            )
            EventManager.add_event_listener(
                event_name="after_val_inference", handler=EvidentialClassificationInferenceHandler(), skip_if_exists=True
            )
            EventManager.add_event_listener(event_name="after_predict", handler=EvidentialClassificationInferenceHandler(), skip_if_exists=True)
        else:
            EventManager.add_event_listener(
                event_name="before_tracker_update", handler=EvidentialRegressionProcessingHandler(), skip_if_exists=True
            )
            EventManager.add_event_listener(event_name="after_val_inference", handler=EvidentialRegressionInferenceHandler(), skip_if_exists=True)
            EventManager.add_event_listener(event_name="after_predict", handler=EvidentialRegressionInferenceHandler(), skip_if_exists=True)

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        loss = self._loss(outputs, labels, epoch=epoch, loss_annealing=self._loss_annealing)

        return loss


class PaddedLoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, padding_value: int = -1, tokenized: bool = False):
        super().__init__()
        self._loss = loss
        self._loss.reduction = "none"
        self._padding_value = padding_value
        self._tokenized = tokenized

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._tokenized:
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1).long()

        mask = labels != self._padding_value
        labels[~mask] = 0

        loss = self._loss(outputs, labels)

        loss = loss.view_as(mask) * mask
        loss = loss.sum() / mask.sum()
        return loss
