from collections import defaultdict
from typing import Union, Dict, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from rdkit import Chem
from torch_geometric.utils import to_networkx

from lib.data.resources import DataPoint, Batch
from lib.visualization.sketchers import GraphSketcher
from vendor.captum.attr import IntegratedGradients


class IntegratedGradientsExplainer:
    """
    Implements the Integrated Gradients explainability technique introduced in https://arxiv.org/abs/1703.01365.
    The core implementation relies on captum: https://github.com/pytorch/captum
    """

    def __init__(self, model: torch.nn.Module, output_path: str):
        self.model = model
        self.model.eval()

        self.sketcher = GraphSketcher(output_path=output_path)

    def explain(self, data: Union[DataPoint, Batch], target: int) -> Dict[int, np.ndarray]:
        graphs = data.inputs["graph"]

        number_nodes = graphs.x.shape[0]
        input_mask = graphs.x.requires_grad_(True).to(graphs.x.device)
        integrated_gradients = IntegratedGradients(self.model_forward)

        mask = integrated_gradients.attribute(
            input_mask, target=target, additional_forward_args=(data,), internal_batch_size=number_nodes
        )

        node_mask = np.abs(mask.cpu().detach().numpy()).sum(axis=1)
        return self.per_mol_mask(data, node_mask)

    def per_mol_mask(self, data: DataPoint, node_masks: np.ndarray) -> Dict[int, np.ndarray]:
        node_mask_per_mol = defaultdict(list)
        batch_ids = data.inputs["graph"].batch

        for node_mask_value, batch_id in zip(node_masks, batch_ids):
            node_mask_per_mol[batch_id.item()].append(node_mask_value)

        for batch_id, node_mask in node_mask_per_mol.items():
            if np.max(node_mask) > 0:
                node_mask_per_mol[batch_id] = (np.array(node_mask) / np.array(node_mask).max())

        return node_mask_per_mol

    def model_forward(self, node_mask: torch.Tensor, data: Union[DataPoint, Batch]) -> torch.Tensor:
        data.inputs["graph"].x = node_mask

        if not hasattr(data.inputs["graph"], "batch"):
            data.inputs["graph"].batch = torch.zeros(
                data.inputs["graph"].x.shape[0], dtype=int
            ).to(data.inputs["graph"].x.device)

        return self.model(data.inputs)

    def to_molecule(self, data: torch_geometric.data.Data) -> nx.Graph:
        mol = Chem.MolFromSmiles(data.smiles)
        g = to_networkx(data, node_attrs=["x"])

        for (u, data), atom in zip(g.nodes(data=True), mol.GetAtoms()):
            data["name"] = atom.GetSymbol()
            del data["x"]

        return g

    def to_mol_list(self, data: DataPoint) -> Tuple[List, List]:
        if isinstance(data.inputs["graph"], torch_geometric.data.Batch):
            data_list = data.inputs["graph"].to_data_list()
            dataset_sample_ids = data.ids
        else:
            data_list = [data.inputs["graph"]]
            dataset_sample_ids = [data.id_]

        return data_list, dataset_sample_ids

    def visualize(self, data: Union[DataPoint, Batch], target: int, save_path: str) -> None:
        node_mask_per_mol = self.explain(data, target)
        data_list, dataset_sample_ids = self.to_mol_list(data)

        for (batch_sample_id, node_mask), dataset_sample_id in zip(node_mask_per_mol.items(), dataset_sample_ids):
            data = data_list[batch_sample_id]
            mol = self.to_molecule(data)

            plt.figure(figsize=(10, 5))
            plt.title(f"Integrated Gradients {data.smiles}")

            self.sketcher.draw(graph=mol, save_path=save_path, node_mask=node_mask)
