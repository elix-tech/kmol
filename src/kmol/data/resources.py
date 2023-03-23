from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional, Iterable

import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater as TorchGeometricCollater
from ..vendor.graphormer import collater


@dataclass
class DataPoint:
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

@dataclass
class LoadedContent:

    dataset: Iterable[Batch]
    samples: int
    batches: int


class AbstractCollater:
    @abstractmethod
    def apply(self, batch: List[DataPoint]) -> Any:
        raise NotImplementedError


class GeneralCollater(AbstractCollater):
    def __init__(self):
        self._collater = TorchGeometricCollater(follow_batch=[], exclude_keys=[])

    def _unpack(self, batch: List[DataPoint]) -> Batch:
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
        inputs = dict(inputs)
        return Batch(ids=ids, labels=batch[0].labels, inputs=inputs, outputs=outputs)

    def apply(self, batch: List[DataPoint]) -> Batch:

        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            batch.inputs[key] = self._collater.collate(values)

        return batch


class GraphormerCollater(GeneralCollater):
    def __init__(self, max_node: int = 512, multi_hop_max_dist: int = 20, spatial_pos_max: int = 20):
        # TODO: automate increment of max_node without requiring pre-setting
        super().__init__()
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def apply(self, batch: List[DataPoint]) -> Batch:
        batch = self._unpack(batch)
        for key, values in batch.inputs.items():
            if key == "ligand":
                batch.inputs[key] = collater.collater(values, self.max_node, self.multi_hop_max_dist, self.spatial_pos_max)
            else:
                batch_dict = self._collater.collate(values)
                batch.inputs[key] = batch_dict

        return batch


class PaddedCollater(GeneralCollater):
    def __init__(self, padded_column):
        super().__init__()
        self.padded_column = padded_column

    def _pad_and_create_mask(self, seqs, dtype=torch.long):
        max_length = max([len(seq) for seq in seqs])
        padded_seqs = [torch.zeros(max_length, dtype=dtype) for _ in range(len(seqs))]
        mask = [torch.zeros(max_length, dtype=torch.bool) for _ in range(len(seqs))]
        for i, seq in enumerate(seqs):
            seq_tensor = seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=dtype)
            padded_seqs[i][:len(seq)] = seq_tensor.clone().detach()
            mask[i][:len(seq)] = 1

        return padded_seqs, mask

    def _unpack(self, batch: List[DataPoint]) -> Batch:
        ids = []
        inputs = defaultdict(list)
        outputs = []

        for entry in batch:
            ids.append(entry.id_)

            for key, value in entry.inputs.items():
                inputs[key].append(value)

            outputs.append(entry.outputs)
        
        inputs_padded = defaultdict(list)
        
        for key, values in inputs.items():
            if key == self.padded_column:
                inputs_padded[key], inputs_padded["mask"] = self._pad_and_create_mask(values)
            else:
                inputs_padded[key] = values

        outputs_padded, _ = self._pad_and_create_mask(outputs, dtype=torch.float)
        outputs_padded = torch.stack(outputs_padded)
        
        inputs_padded = dict(inputs_padded)

        return Batch(ids=ids, labels=batch[0].labels, inputs=inputs_padded, outputs=outputs_padded)
