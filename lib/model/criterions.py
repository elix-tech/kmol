import torch
from typing import List


class MaskedLoss(torch.nn.Module):

    def __init__(self, loss: torch.nn.modules.loss._Loss):
        super().__init__()

        self._loss = loss
        self._loss.reduction = "none"

    def _handle_missing_values(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        loss *= mask
        loss = loss.sum(dim=0) / (mask.sum(dim=0) + 1e-8)

        return loss

    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:

        mask = ground_truth == ground_truth
        ground_truth[~mask] = 0

        loss = self._loss(logits, ground_truth)
        loss = self._handle_missing_values(loss, mask)

        loss = loss.mean()
        return loss


class MultiTaskLoss(torch.nn.Module):

    def __init__(
            self, regression_loss: torch.nn.Module, classification_loss: torch.nn.Module,
            regression_tasks: List[int], classification_tasks: List[int],
            regression_weight: float = 1, classification_weight: float = 1
    ):
        super().__init__()

        self._regression_loss = regression_loss
        self._classification_loss = classification_loss

        self._regression_tasks = regression_tasks
        self._classification_tasks = classification_tasks

        self._regression_weight = regression_weight
        self._classification_weight = classification_weight

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        regression_loss = self._regression_loss(
            outputs[:, self._regression_tasks],
            labels[:, self._regression_tasks]
        )

        classification_loss = self._classification_loss(
            outputs[:, self._classification_tasks],
            labels[:, self._classification_tasks]
        )

        return regression_loss * self._regression_weight + classification_loss * self._classification_weight
