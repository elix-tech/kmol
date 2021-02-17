from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data.dataloader import Collater as TorchGeometricCollater


@dataclass
class Data:
    id_: Optional[Union[str, int]] = None
    labels: Optional[List[str]] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Union[List[Any], np.ndarray]] = None


@dataclass
class Batch:
    ids: List[Union[str, int]]
    labels: List[str]
    inputs: Dict[str, torch.Tensor]
    outputs: torch.FloatTensor


class Collater:

    def __init__(self, device: torch.device):
        self._device = device
        self._collater = TorchGeometricCollater(follow_batch=[])

    def _unpack(self, batch: List[Data]) -> Batch:
        ids = []
        inputs = defaultdict(list)
        outputs = []

        for entry in batch:
            ids.append(entry.id_)

            for key, value in entry.inputs.items():
                inputs[key].append(value)

            outputs.append(entry.outputs)

        outputs = np.array(outputs)
        outputs = torch.FloatTensor(outputs)

        return Batch(ids=ids, labels=batch[0].labels, inputs=inputs, outputs=outputs)

    def _set_device(self, batch: Batch) -> None:
        batch.outputs = batch.outputs.to(self._device)
        for key, values in batch.inputs.items():
            batch.inputs[key] = values.to(self._device)

    def _collate(self, data: Sequence[Any]) -> Any:
        try:
            return self._collater.collate(data)
        except TypeError as e:
            element = data[0]

            if isinstance(element, np.ndarray):
                return torch.tensor(data, dtype=torch.float)

            raise e

    def apply(self, batch: List[Data]) -> Batch:

        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            batch.inputs[key] = self._collate(data=values)

        self._set_device(batch)
        return batch
