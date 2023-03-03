
import torch
from torch_geometric.data import Batch
from torch.nn.parallel.scatter_gather import is_namedtuple
from torch.nn.parallel._functions import Scatter


class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module, name)
    
    def scatter(self, inputs, kwargs, device_ids, dim=0):
        r"""Scatter with support for kwargs dictionary"""
        inputs = self._scatter(inputs, device_ids, dim) if inputs else []
        kwargs = self._scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
        elif len(kwargs) < len(inputs):
            kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs


    def _scatter(self, inputs, target_gpus, dim=0):
        r"""
        Slices tensors into approximately equal chunks and
        distributes them across given GPUs. Duplicates
        references to objects that are not tensors.
        """
        def scatter_map(obj):
            if isinstance(obj, torch.Tensor):
                return Scatter.apply(target_gpus, None, dim, obj)
            if isinstance(obj, Batch):
                return self.scatter_graph(obj.to_data_list(), target_gpus)
            if is_namedtuple(obj):
                return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(scatter_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                return [list(i) for i in zip(*map(scatter_map, obj))]
            if isinstance(obj, dict) and len(obj) > 0:
                return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
            return [obj for targets in target_gpus]

        # After scatter_map is called, a scatter_map cell will exist. This cell
        # has a reference to the actual function scatter_map, which has references
        # to a closure that has a reference to the scatter_map cell (because the
        # fn is recursive). To avoid this reference cycle, we set the function to
        # None, clearing the cell
        try:
            res = scatter_map(inputs)
        finally:
            scatter_map = None
        return res


    def scatter_graph(self, data_list, device_ids):
        num_devices = min(len(device_ids), len(data_list))

        count = torch.tensor([data.num_nodes for data in data_list])
        cumsum = count.cumsum(0)
        cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
        device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
        device_id = (device_id[:-1] + device_id[1:]) / 2.0
        device_id = device_id.to(torch.long)  # round.
        split = device_id.bincount().cumsum(0)
        split = torch.cat([split.new_zeros(1), split], dim=0)
        split = torch.unique(split, sorted=True)
        split = split.tolist()

        return [
            Batch.from_data_list(data_list[split[i]:split[i + 1]],
                                #  follow_batch=self.follow_batch,
                                #  exclude_keys=self.exclude_keys
                                 ).to(
                                     torch.device('cuda:{}'.format(
                                         device_ids[i])))
            for i in range(len(split) - 1)
        ]