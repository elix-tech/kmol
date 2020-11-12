from collections import OrderedDict
from typing import List, Dict

import torch

from mila.factories import AbstractAggregator


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
            owner = checkpoint_path.split("/")[-1].split(".")[0]
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
