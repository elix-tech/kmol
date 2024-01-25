import os
from collections import defaultdict
from typing import Union, Dict, Tuple, List, TextIO

import numpy as np
import torch
import torch_geometric
from captum.attr import IntegratedGradients

from kmol.visualization.sketchers import AbstractSketcher
from kmol.core.config import Config
from kmol.core.helpers import SuperFactory, Loggable
from kmol.data.resources import DataPoint, Batch


class IntegratedGradientsExplainer(Loggable):
    """
    Implements the Integrated Gradients explainability technique introduced in https://arxiv.org/abs/1703.01365.
    The core implementation relies on captum: https://github.com/pytorch/captum
    """

    def __init__(
        self, model: torch.nn.Module, config: Config, is_binary_classification: bool = False, is_multitask: bool = False
    ):
        Loggable.__init__(self, config.visualizer["mapping_file_path"])
        self.log("file_path,smiles\n")

        self.model = model
        self.model.eval()
        self.graph_input_key = config.visualizer.get("graph_input_key", "graph")
        self.sketcher = SuperFactory.create(AbstractSketcher, config.visualizer["sketcher"])
        self.is_binary_classification = is_binary_classification
        self.is_multitask = is_multitask

    def _create_logger(self, log_file_path: str) -> TextIO:
        if "/" in log_file_path:
            mapping_file_location = log_file_path.rsplit("/", 1)[0]
            os.makedirs(mapping_file_location, exist_ok=True)

        return open(log_file_path, "w")

    def explain(self, data: Union[DataPoint, Batch], target: int) -> Dict[int, np.ndarray]:
        graphs = data.inputs[self.graph_input_key]

        number_nodes = graphs.x.shape[0]
        input_mask = graphs.x.requires_grad_(True).to(graphs.x.device)
        integrated_gradients = IntegratedGradients(self.model_forward)
        mask = integrated_gradients.attribute(
            input_mask,
            target=target,
            additional_forward_args=(data,),
            internal_batch_size=number_nodes,
        )

        node_mask = np.abs(mask.cpu().detach().numpy()).sum(axis=1)
        return self.per_mol_mask(data, node_mask)

    def per_mol_mask(self, data: DataPoint, node_masks: np.ndarray) -> Dict[int, np.ndarray]:
        node_mask_per_mol = defaultdict(list)
        batch_ids = data.inputs[self.graph_input_key].batch

        for node_mask_value, batch_id in zip(node_masks, batch_ids):
            node_mask_per_mol[batch_id.item()].append(node_mask_value)

        for batch_id, node_mask in node_mask_per_mol.items():
            if np.max(node_mask) > 0:
                node_mask_per_mol[batch_id] = np.array(node_mask) / np.array(node_mask).max()

        return node_mask_per_mol

    def model_forward(self, node_mask: torch.Tensor, data: Union[DataPoint, Batch]) -> torch.Tensor:
        data.inputs[self.graph_input_key].x = node_mask

        if not hasattr(data.inputs[self.graph_input_key], "batch"):
            data.inputs[self.graph_input_key].batch = torch.zeros(
                data.inputs[self.graph_input_key].x.shape[0], dtype=int
            ).to(data.inputs[self.graph_input_key].x.device)

        return self.model(data.inputs)

    def model_predict(self, data):
        with torch.no_grad():
            output = self.model(data.inputs)

            if self.is_binary_classification:
                class_probs = torch.sigmoid(output)
                return class_probs
            else:
                return output

    def to_mol_list(self, data: DataPoint) -> Tuple[List, List]:
        if isinstance(data.inputs[self.graph_input_key], torch_geometric.data.Batch):
            data_list = data.inputs[self.graph_input_key].to_data_list()
            dataset_sample_ids = data.ids
        else:
            data_list = [data.inputs[self.graph_input_key]]
            dataset_sample_ids = [data.id_]

        labels = data.outputs
        return data_list, dataset_sample_ids, labels

    def visualize(self, data: Union[DataPoint, Batch], target: int, save_path: str) -> None:
        if self.is_multitask:
            protein_target = np.argwhere(data.outputs.cpu().numpy() == data.outputs.cpu().numpy())

        node_mask_per_mol = self.explain(data, target)
        preds_ten = self.model_predict(data)
        data_list, dataset_sample_ids, labels = self.to_mol_list(data)

        preds_arr, labels_arr = self._gen_arr_from_ten(preds_ten, labels)

        if self.is_multitask:
            preds_arr = preds_arr[:, protein_target[:, 1]].squeeze(axis=-1)
            labels_arr = labels_arr[:, protein_target[:, 1]].squeeze(axis=-1)

        if self.is_binary_classification:
            preds_arr = np.where(preds_arr >= 0.5, 1.0, 0.0)

        preds_arr = preds_arr.tolist()
        labels_arr = labels_arr.tolist()

        for (batch_sample_id, node_mask), dataset_sample_id in zip(node_mask_per_mol.items(), dataset_sample_ids):
            sample = data_list[batch_sample_id]

            prediction = preds_arr[batch_sample_id]
            label = labels_arr[batch_sample_id]
            if not self.is_binary_classification:
                prediction = round(prediction, 3)
                label = round(label, 3)

            self.sketcher.draw(sample, save_path, node_mask, prediction, label)
            self.log("{},{}\n".format(save_path, sample.smiles))

    def _gen_arr_from_ten(self, predictons_ten, labels_ten):
        pred_arr = predictons_ten.cpu().detach().numpy()
        label_arr = labels_ten.cpu().detach().numpy()

        if self.is_binary_classification:
            if not self.is_multitask:
                pred_arr = pred_arr.squeeze(axis=-1)
                label_arr = label_arr.squeeze(axis=-1)

        return pred_arr, label_arr
