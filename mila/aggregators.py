import collections
from typing import List

import torch

from mila.factories import AbstractAggregator


class TorchAggregator(AbstractAggregator):

    def run(self, checkpoint_paths: List[str], save_path: str) -> None:
        output = None

        for checkpoint_path in checkpoint_paths:
            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model: collections.OrderedDict = state["model"]

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
