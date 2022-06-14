from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.nn import (GlobalAttention, Set2Set, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class MaxReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_max_pool(x, batch)


class SumReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_add_pool(x, batch)


class MeanReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_mean_pool(x, batch)


class AttentionReadOut(GlobalAttention):
    def __init__(self, in_channels: int, full: bool = True, out_channels: Optional[int] = None, **kwargs):
        """
        When full is set to true, attention is computed separately on each feature channel.
        """
        out = in_channels if out_channels is None else out_channels
        self.attention_out_dim = out if full else 1
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, self.attention_out_dim),
        )
        nn = torch.nn.Linear(in_channels, out_channels) if out_channels is not None else None
        super().__init__(gate_nn, nn)
        self.out_dim = out

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        size = int(batch.max().item() + 1)
        if self.attention_out_dim == 1:
            return super().forward(x, batch, size=size)
        else:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            gate = self.gate_nn(x).view(-1, self.attention_out_dim)
            x = self.nn(x) if self.nn is not None else x
            if not gate.dim() == x.dim() and gate.size(0) == x.size(0):
                raise ValueError(f"Wrong input dimension: {gate.shape}, {x.shape}")

            gate = softmax(gate, batch, num_nodes=size)
            out = scatter_add(gate * x, batch, dim=0, dim_size=size)

            return out


class MLPSumReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, **kwargs):
        super().__init__()
        out = out_channels if out_channels is not None else in_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, out),
        )
        self.out_dim = out

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        x = self.mlp(x)
        return global_add_pool(x, batch)


class Set2SetReadOut(Set2Set):
    def __init__(self, in_channels: int, processing_steps=4, num_layers=2, **kwargs):
        super().__init__(in_channels, processing_steps, num_layers)
        self.out_dim = 2 * in_channels


class CombinedReadOut(torch.nn.Module):
    def __init__(self, read_out_list: Union[Tuple[str, ...], List[str]], read_out_kwargs: dict):
        super().__init__()
        self.read_outs = torch.nn.ModuleList(
            [get_read_out(f, read_out_kwargs) for f in read_out_list]
        )
        self.out_dim = sum([read_out.out_dim for read_out in self.read_outs])

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return torch.cat([read_out(x, batch) for read_out in self.read_outs], dim=1)


READOUT_FUNCTIONS = {
    "max": MaxReadOut,
    "sum": SumReadOut,
    "mean": MeanReadOut,
    "set2set": Set2SetReadOut,
    "attention": AttentionReadOut,
    "mlp_sum": MLPSumReadOut,
}


def get_read_out(read_out: Union[str, Tuple[str, ...], List[str]], read_out_kwargs: Dict):
    if "in_channels" not in read_out_kwargs:
        raise ValueError("Can't instantiate read_out without `in_channels` argument")
    if isinstance(read_out, tuple) or isinstance(read_out, list):
        return CombinedReadOut(read_out, read_out_kwargs)
    else:
        read_out_fn = READOUT_FUNCTIONS.get(read_out, None)
        if read_out_fn is None:
            raise ValueError(f"Unknown read_out function : {read_out}")
        return read_out_fn(**read_out_kwargs)
