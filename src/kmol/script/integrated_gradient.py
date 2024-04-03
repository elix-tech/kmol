from typing import Dict
import pandas as pd
from pathlib import Path
import inspect

import torch
from tqdm import tqdm
import numpy as np
from captum.attr import Attribution as AbstractAttribution
from torch_geometric.data import Data as TorchGeometricData
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing

from mila.factories import AbstractScript

from kmol.core.config import Config
from kmol.model.executors import Predictor
from kmol.core.helpers import SuperFactory
from kmol.data.streamers import GeneralStreamer
from kmol.data.loaders import AbstractLoader
from kmol.model.architectures.abstract_network import AbstractNetwork
from kmol.model.architectures import GraphConvolutionalNetwork

from kmol.core.logger import LOGGER as logging

GRAPH_CAPTUM_FEAT = ["x", "molecule_features", "edge_attr"]


class CaptumScript(AbstractScript):
    def __init__(self, attribution, config_path, reduction, n_steps: int = 50) -> None:
        super().__init__()
        self._config = Config.from_file(config_path, job_command="eval")
        self.model = SuperFactory.create(AbstractNetwork, self._config.model)
        self.tmp_executor = Predictor(self._config)
        self.model = self.tmp_executor.network
        self.filter_captum_feature()
        self.loader = SuperFactory.create(AbstractLoader, self._config.loader)
        self.column_of_interest = self.loader._input_columns + self.loader._target_columns
        streamer = GeneralStreamer(config=self._config)
        self.data_loader = iter(
            streamer.get(
                split_name="test",
                batch_size=1,
                shuffle=False,
                mode=GeneralStreamer.Mode.TEST,
            ).dataset
        )
        self.dataset = self.loader._dataset
        self.reduction = reduction
        self.n_steps = n_steps
        if not reduction in ["sum", "mean"]:
            raise ValueError(f"{reduction} must be in ['sum', 'mean']")
        self.attribution = attribution

    def filter_captum_feature(self):
        global GRAPH_CAPTUM_FEAT
        input_parameters = []
        for module in self.model.modules():
            if issubclass(type(module), MessagePassing):
                input_parameters += list(inspect.signature(module.forward).parameters.keys())
            if isinstance(module, GraphConvolutionalNetwork):
                if issubclass(type(module.molecular_head), torch.nn.Module):
                    input_parameters += ["molecule_features"]
        unused_params = set(GRAPH_CAPTUM_FEAT) - set(input_parameters)
        GRAPH_CAPTUM_FEAT = [feat for feat in GRAPH_CAPTUM_FEAT if feat not in unused_params]

    def run(self):
        self.attribution_innit()
        results = pd.DataFrame([], columns=self.column_of_interest)
        attributions = {i: {} for i in range(self.model.out_features)}
        for data in tqdm(self.data_loader):
            for i in range(self.model.out_features):
                attributions = self.compute_attribute(data, attributions, target=i)
            results = pd.concat([results, self.dataset.loc[data.ids, self.column_of_interest]])
        self.update_df(results, attributions)

        results.to_csv(Path(self._config.output_path) / "captum_results.csv")

    def attribution_innit(self):
        model_wrapper = ModelWrapper(self.model, self.attribution["type"], self.n_steps)
        model_wrapper.to(self._config.get_device())
        self.attributor = CustomCaptum(self.attribution, model_wrapper)

    def compute_attribute(self, data, attributions, target):
        self.tmp_executor._to_device(data)
        try:
            attribute_dict = self.attributor.attribute(data.inputs, target=target, n_steps=self.n_steps)
            attribute_dict = self.regroup_results(attribute_dict)
            attributions = self.update_dict(attributions, attribute_dict, target)
        except Exception as e:
            logging.info(f"One element has been skipped, {e}")
            attributions = self.update_dict(
                attributions, dict(zip(self.attributor.output_name, [0] * len(self.attributor.output_name))), target
            )
        return attributions

    def update_dict(self, attributions, update, target):
        for k, v in update.items():
            list_attr = attributions[target].get(k, [])
            list_attr.append(v)
            attributions[target][k] = list_attr
        return attributions

    def update_df(self, results, attributions):
        for target, target_attributions in attributions.items():
            for name, attribution in target_attributions.items():
                results[f"{target}_{name}"] = attribution
        return results

    def regroup_results(self, attribution_results: Dict[str, torch.Tensor]):
        reduce_attr = {name: 0 for name in self.attributor.output_name}
        for name, attribution in zip(self.attributor.output_name, attribution_results):
            if name == "molecule_features":
                reduce_attr[name] = self.apply_reduction(attribution)
            elif name in GRAPH_CAPTUM_FEAT:
                reduce_attr[name] += self.apply_reduction(attribution)
            else:
                reduce_attr[name] = self.apply_reduction(attribution)
        return reduce_attr

    def apply_reduction(self, attribution: torch.Tensor):
        if self.reduction == "sum":
            return attribution.sum().cpu().detach().numpy().item()
        elif self.reduction == "mean":
            return attribution.mean().cpu().detach().numpy().item()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, attribution_type, n_steps):
        super(ModelWrapper, self).__init__()
        self.model: AbstractNetwork = model
        self.attribution_type = attribution_type
        self.n_steps = n_steps

    def forward(self, *args):
        input_dict = self.tensor_to_dict(args)
        return self.model(input_dict)

    def tensor_to_dict(self, inputs):
        inputs = list(inputs)
        original_feature_key = inputs.pop(-1)
        data = {}
        id_input = 0
        for key_name, _type in original_feature_key["inputs"]:
            if issubclass(_type, TorchGeometricData):
                id_input, data[key_name] = self.process_graph_feature(id_input, inputs, original_feature_key["graph_attr"])
            else:
                data[key_name] = inputs[id_input]
                id_input += 1
        return data

    def process_graph_feature(self, id_input, inputs, graph_attr):
        n_graph_feat = len(graph_attr)
        # data_graph = {"x": inputs[id_input]}

        # graph feature to evaluate with capt are located at the end.
        captum_graph_attr = []
        for graph_key in graph_attr:
            if graph_key in GRAPH_CAPTUM_FEAT:
                captum_graph_attr.append(inputs[id_input])
                n_graph_feat = n_graph_feat - 1
                id_input += 1

        data_graph = list(inputs[-n_graph_feat:]) + captum_graph_attr
        if data_graph[0].size(0) % self.n_steps == 0:
            data_graph = [reshape_with_extra_dim(t, self.n_steps) if isinstance(t, torch.Tensor) else t for t in data_graph]
        data_graph = dict(zip(graph_attr, data_graph))
        if list(data_graph.values())[0].size(0) % self.n_steps == 0:
            return id_input, Batch.from_data_list(
                [TorchGeometricData(**get_ith_elements(data_graph, i)) for i in range(self.n_steps)]
            )
        else:
            return id_input, Batch.from_data_list([TorchGeometricData(**data_graph)])


def get_ith_elements(dic, i):
    return {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in dic.items()}


def reshape_with_extra_dim(tensor, n_steps):
    assert tensor.size(0) % n_steps == 0, f"First dimension size should be divisible by {n_steps}"

    new_shape = [n_steps, tensor.size(0) // n_steps] + list(tensor.shape[1:])
    return tensor.reshape(new_shape)


class CustomCaptum:
    def __init__(self, attribution: AbstractAttribution, forward_func, reduction="sum") -> None:
        attribution = attribution | {"forward_func": forward_func}
        if "LayerIntegratedGradients" in attribution["type"]:
            if isinstance(attribution["layer"], list):
                attribution["layer"] = [forward_func.get_submodule(f"model.{l}") for l in attribution["layer"]]
            else:
                attribution["layer"] = forward_func.get_submodule(f"model.{attribution['layer']}")
        self.attribution = SuperFactory.create(AbstractAttribution, attribution)
        self.output_name = []

    def attribute(self, inputs, n_steps=50, **kwargs):
        inputs_tensor, additional_attr = self.dict_to_tensor(inputs)
        results = self.attribution.attribute(inputs_tensor, additional_forward_args=additional_attr, n_steps=n_steps, **kwargs)
        return results

    def dict_to_tensor(self, data_inputs):
        inputs, output_name, additional_attr = [], [], []
        original_feature_key = {"inputs": [], "graph_attr": []}
        for key, _input in data_inputs.items():
            if isinstance(_input, TorchGeometricData):
                graph_inputs, additional_attr, original_feature_key["graph_attr"] = self.process_graph_feature(_input)
                output_name += [f"{key}_{name}" for name in original_feature_key["graph_attr"][-len(graph_inputs) :]]
                inputs += graph_inputs
            elif isinstance(_input, torch.Tensor):
                inputs.append(_input)
                output_name += [key]
            else:
                continue
            original_feature_key["inputs"].append((key, type(_input)))
        self.output_name = output_name
        return tuple(inputs), tuple(additional_attr + [original_feature_key])

    def process_graph_feature(self, graph_data: TorchGeometricData):
        graph_input, additional_attr = [], []
        graph_attr = list(graph_data.to_dict().keys())
        for graph_key, v in graph_data.items():
            if graph_key in GRAPH_CAPTUM_FEAT:
                graph_input.append(v)
                i_mol_feat = np.where(np.isin(graph_attr, [graph_key]))[0][0]
                graph_attr.append(graph_attr.pop(i_mol_feat))
            else:
                additional_attr.append(v)
        return list(graph_input), additional_attr, graph_attr
