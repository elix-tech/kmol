from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional

import torch
from torch_geometric.data.dataloader import Collater as TorchGeometricCollater
import numpy as np


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
    inputs: Dict[str, List[Any]]
    outputs: torch.FloatTensor


class Collater:

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

    def apply(self, batch: List[Data]) -> Batch:

        batch = self._unpack(batch)

        collater = TorchGeometricCollater(follow_batch=[])
        for key, values in batch.inputs.items():
            batch.inputs[key] = collater.collate(values)

        return batch
