from dataclasses import dataclass

import torch


@dataclass
class ProteinLigandBatch:
    labels: torch.Tensor
    ligand_features: torch.Tensor
    protein_features: torch.Tensor
