import torch
from typing import List, Dict

from .evidential_losses import (
    edl_mse_loss,
    edl_log_loss,
    edl_digamma_loss,
    edl_reg_log,
    edl_log_loss_multilabel_logits,
    edl_log_loss_multilabel_nologits,
    edl_log_loss_multilabel_nologits_masked_weighted,
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
        super().__init__()

        self._loss_type = loss["type"]

        loss_dict = {
            "class_log": edl_log_loss,
            "class_log_multilabel_logits": edl_log_loss_multilabel_logits,
            "class_log_multilabel_nologits": edl_log_loss_multilabel_nologits,
            "class_mse": edl_mse_loss,
            "class_digamma": edl_digamma_loss,
            "reg_log": edl_reg_log,
            "class_log_multilabel_nologits_masked": edl_log_loss_multilabel_nologits_masked_weighted,
        }
        self._loss = loss_dict[loss["type"]]

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self._loss(outputs, labels)

        return loss
