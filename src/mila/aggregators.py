import importlib
from collections import OrderedDict
from copy import deepcopy
from typing import List, Dict, Type, Any

import numpy as np
import torch

from .factories import AbstractAggregator, AbstractConfiguration, AbstractExecutor


class PlainTorchAggregator(AbstractAggregator):

    def run(self, checkpoint_paths: List[str], save_path: str) -> None:
        output = None

        for checkpoint_path in checkpoint_paths:
            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model: OrderedDict = state["model"]

            if output is None:
                output = model
                continue

            for key, value in model.items():
                output[key] += value

        checkpoints_count = len(checkpoint_paths)
        for key, value in output.items():
            if value.is_floating_point():
                output[key] = torch.div(value, checkpoints_count)
            else:
                output[key] = torch.floor_divide(value, checkpoints_count)

        output = {"model": output}
        torch.save(output, save_path)


class WeightedTorchAggregator(AbstractAggregator):

    def __init__(self, weights: Dict[str, float]):
        self._weights = weights

    def run(self, checkpoint_paths: List[str], save_path: str) -> None:
        output = OrderedDict()

        for checkpoint_path in checkpoint_paths:
            owner = str(checkpoint_path).split("/")[-1].split(".")[0]
            weight = self._weights[owner]

            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model: OrderedDict = state["model"]

            for key, value in model.items():
                if key not in output:
                    output[key] = value * weight
                else:
                    output[key] += value * weight

        output = {"model": output}
        torch.save(output, save_path)


class BenchmarkedTorchAggregator(AbstractAggregator):

    def __init__(
            self, config_type: str, config_path: str,
            executor_type: str, target_metric: str, minimized_metric: bool = False
    ):
        self._config_instantiator: Type[AbstractConfiguration] = self._reflect(config_type)
        self._executor_instantiator: Type[AbstractExecutor] = self._reflect(executor_type)
        self._config = self._config_instantiator.from_file(config_path)
        self._target_metric = target_metric
        self._minimized_metric = minimized_metric

    def _reflect(self, dependency: str) -> Type[Any]:
        module_name, class_name = dependency.rsplit(".", 1)
        module = importlib.import_module(module_name)

        return getattr(module, class_name)

    def run(self, checkpoint_paths: List[str], save_path: str) -> None:
        output = OrderedDict()

        weights = []
        for checkpoint_path in checkpoint_paths:
            config = deepcopy(self._config)
            config.checkpoint_path = checkpoint_path

            executor = self._executor_instantiator(config=config)
            results = executor.eval()

            weights.append(np.mean(getattr(results, self._target_metric)))

        weights = np.array(weights) / sum(weights)
        if self._minimized_metric:
            weights = 1 - weights

        for client_id, checkpoint_path in enumerate(checkpoint_paths):
            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model: OrderedDict = state["model"]

            for key, value in model.items():
                if key not in output:
                    output[key] = value * weights[client_id]
                else:
                    output[key] += value * weights[client_id]

        output = {"model": output}
        torch.save(output, save_path)
