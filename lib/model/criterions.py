from typing import Optional

import torch


class WeightedBinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            input=input, target=target, weight=weight,
            pos_weight=self.pos_weight, reduction=self.reduction
        )
