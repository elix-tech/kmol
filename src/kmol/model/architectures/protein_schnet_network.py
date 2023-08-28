from typing import Callable, Optional, Any, Union, Dict, List
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import SchNet, InteractionBlock, CFConv
from torch_geometric.typing import OptTensor
from math import pi as PI

from torch.nn import Sequential, Linear, ReLU, Embedding, ModuleList
from torch import Tensor
import torch

from .abstract_network import AbstractNetwork
from ..read_out import get_read_out


class ProteinSchnetNetwork(AbstractNetwork, SchNet):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_feature: int = 1,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10,
        interaction_graph: Union[Callable[..., Any], None] = None,
        max_num_neighbors: int = 32,
        readout: str = "sum",
        dipole: bool = False,
        mean: Union[float, None] = None,
        std: Union[float, None] = None,
        atomref: OptTensor = None,
        num_lp_interactions: int = 52,
    ):
        """
        See pytorch documentation for base parameters.
        num_lp_interactions: Number of possible interaction
        """
        super().__init__(
            hidden_channels,
            num_filters,
            num_interactions,
            num_gaussians,
            cutoff,
            interaction_graph,
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )

        self.protein_embedding = Embedding(100, hidden_channels, padding_idx=0)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = ProteinInteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff, num_lp_interactions)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_feature)
        self.out_feature = out_feature

        readout_kwargs = {}
        readout_kwargs.update({"in_channels": hidden_channels // 2})
        self.readout = get_read_out(readout, readout_kwargs)

    def get_requirements(self) -> List[str]:
        return ["schnet_inputs"]

    def forward(self, data: Dict[str, Any]) -> Tensor:
        r"""
        Args:
            z (LongTensor): Atomic number of each atom with shape

            pos (Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            protein_mask (Tensor): Mask of atom corresponding to protein atoms
                with shape :obj:`[num_atoms]`.
            z_protein : Atomic number of each protein with shape
                :obj:`[num_atoms]`.
            lp_edge_[]:  Edge information of the protein ligand interaction,
                both lp_edge_index and lp_edge_attr need to be present.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`),
        """
        z, pos, protein_mask, lp_edge_index, lp_edge_attr, batch = data[self.get_requirements()[0]]
        batch = torch.zeros_like(z) if batch is None else batch
        self.has_lp_interaction = protein_mask.sum() > 0 and lp_edge_index is not None

        h = torch.Tensor(z.shape[0], self.hidden_channels)
        h[~protein_mask] = self.embedding(z[~protein_mask])
        h[protein_mask] = self.protein_embedding(z[protein_mask])
        # Create a mask so that protein and atoms are considered a different molecule.
        # There won't be bond between a protein and ligand outside of the
        mask_ligand_protein_batch = batch.clone()
        mask_ligand_protein_batch[protein_mask] += torch.max(batch) + 1

        edge_index, edge_weight = self.interaction_graph(pos, mask_ligand_protein_batch)
        edge_attr = self.distance_expansion(edge_weight)

        edge_index = torch.hstack([edge_index, lp_edge_index]).long()

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr, lp_edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch).squeeze(0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out


class ProteinInteractionBlock(InteractionBlock):
    def __init__(self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float, num_lp_interactions: int):
        super().__init__(hidden_channels, num_gaussians, num_filters, cutoff)
        self.mlp_protein = Sequential(
            Linear(num_lp_interactions, num_filters),
            ReLU(),
            Linear(num_filters, num_filters),
        )
        self.conv = ProteinCFConv(hidden_channels, hidden_channels, num_filters, self.mlp, self.mlp_protein, cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "mlp_protein"):
            torch.nn.init.xavier_uniform_(self.mlp_protein[0].weight)
            self.mlp_protein[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.mlp_protein[2].weight)
            self.mlp_protein[2].bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor, lp_edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, lp_edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class ProteinCFConv(MessagePassing):
    def __init__(
        self, in_channels: int, out_channels: int, num_filters: int, nn_ligand: Sequential, nn_protein, cutoff: float
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn_ligand = nn_ligand
        self.nn_protein = nn_protein
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def compute_weight_edge_ligand(self, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        return self.nn_ligand(edge_attr) * C.view(-1, 1)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor, lp_edge_attr: Tensor) -> Tensor:
        w_ligand = self.compute_weight_edge_ligand(edge_weight, edge_attr)
        w_protein = self.nn_protein(lp_edge_attr)
        W = torch.vstack([w_ligand, w_protein])
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


if __name__ == "__main__":
    import pickle

    with open("test_data.pkl", "rb") as file:
        data = pickle.load(file)

    model = ProteinSchnet(out_feature=128)
    r = model(
        torch.hstack([torch.Tensor(data["z"]), torch.Tensor(data["z_protein"])]).long(),
        torch.vstack([torch.Tensor(data["pos"]), torch.Tensor(data["pos_protein"])]).squeeze(),
        protein_mask=torch.hstack([torch.zeros(len(data["z"])), torch.ones(len(data["z_protein"]))]).bool(),
        lp_edge_index=torch.Tensor(data["lp_edge_index"].tolist()).T,
        lp_edge_attr=torch.Tensor(data["lp_edge_feature"].tolist()),
    )
    a = 0
