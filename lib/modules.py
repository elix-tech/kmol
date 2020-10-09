from typing import Optional

import torch
from ogb.graphproppred.mol_encoder import full_atom_feature_dims


class AtomEncoder(torch.nn.Module):

    def __init__(self, hidden_channels: int):
        super().__init__()

        self.embeddings = torch.nn.ModuleList()
        for _ in full_atom_feature_dims:
            self.embeddings.append(torch.nn.Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = 0

        for i in range(x.size(1)):
            output += self.embeddings[i](x[:, i])

        return output


class WeightedBinaryCrossEntropyLoss(torch.nn.BCEWithLogitsLoss):

    def forward(
            self, predictions: torch.Tensor, ground_truth: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        return torch.nn.functional.binary_cross_entropy_with_logits(
            input=predictions, target=ground_truth, weight=weights,
            pos_weight=self.pos_weight, reduction=self.reduction
        )
