from typing import Optional

import torch


class WeightedBinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):

    def forward(
            self, predictions: torch.Tensor, ground_truth: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        return torch.nn.functional.binary_cross_entropy_with_logits(
            input=predictions, target=ground_truth, weight=weights,
            pos_weight=self.pos_weight, reduction=self.reduction
        )
