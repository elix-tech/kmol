from dataclasses import dataclass

import torch


@dataclass
class ProteinLigandBatch:
    y: torch.Tensor
    ligand_features: torch.Tensor
    protein_features: torch.Tensor
