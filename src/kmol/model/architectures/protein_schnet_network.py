from typing import Callable, Optional, Any, Union, Dict, List
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import SchNet, InteractionBlock, CFConv
from torch_geometric.typing import OptTensor
from math import pi as PI

from torch.nn import Sequential, Linear, ReLU, Embedding, ModuleList
from torch import Tensor
import torch

from kmol.model.architectures.abstract_network import AbstractNetwork
from kmol.model.read_out import get_read_out


class ProteinSchnetNetwork(AbstractNetwork, SchNet):
    """
    Model to leverage 3D data of complex file, it uses both protein information
    in the surrounding of the ligand and interaction information extracted
    with intDesc.
    It's expected to be used with the AtomTypeExtensionPdbFeaturizer and IntdescFeaturizer
    """

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
        num_atomtype_ligand: int = 1,
        num_atomtype_protein: int = 1,
    ):
        """
        See pytorch documentation for base parameters.
        num_lp_interactions: Number of possible interaction
        num_atomtype_ligand: Number should match config in the IntDescFeaturizer
        num_atomtype_protein: Number should match config in the IntDescFeaturizer
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
        embedding_channel = hidden_channels // num_atomtype_ligand
        protein_embedding_channel = hidden_channels // num_atomtype_protein
        if (
            embedding_channel * num_atomtype_ligand != hidden_channels
            or protein_embedding_channel * num_atomtype_protein != hidden_channels
        ):
            raise ValueError("The hidden channel needs to be divisible by num_atomtype_ligand and num_atomtype_protein")

        self.embedding = ModuleList([Embedding(100, embedding_channel, padding_idx=0) for i in range(num_atomtype_ligand)])
        self.protein_embedding = ModuleList(
            [Embedding(100, protein_embedding_channel, padding_idx=0) for i in range(num_atomtype_protein)]
        )
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = ProteinInteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff, num_lp_interactions)
            self.interactions.append(block)

        self.out_features = out_feature
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, self.out_features)

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
        data_batch = data[self.get_requirements()[0]]
        z = data_batch.z
        z_protein = data_batch.z_protein
        pos = data_batch.coords
        protein_mask = data_batch.protein_mask
        lp_edge_index = data_batch.edge_index
        lp_edge_attr = data_batch.edge_attr
        batch = torch.zeros_like(z) if data_batch.get("batch") is None else data_batch.get("batch")
        self.has_lp_interaction = protein_mask.sum() > 0 and lp_edge_index is not None

        h = torch.zeros([len(batch), self.hidden_channels], device=z.device)
        h[~protein_mask] = torch.hstack([emb(i) for emb, i in zip(self.embedding, z.T)])
        h[protein_mask] = torch.hstack([emb(i) for emb, i in zip(self.protein_embedding, z_protein.T)])

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_index, edge_weight, mask_lp_edge_index = self.get_mask_lp_indice(edge_index, edge_weight, lp_edge_index)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr, mask_lp_edge_index, lp_edge_attr)

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

        if len(out.shape) == 1:
            # In case the last batch is of size 1
            return out.unsqueeze(0)
        else:
            return out

    def get_mask_lp_indice(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, lp_interaction_edge: torch.Tensor):
        """
        Retrieve the indice of the lp edge in the distance edge_index and update it
        if some are missing.
        """
        matching_indices = torch.tensor([], device=edge_index.device, dtype=torch.long)

        for edge in lp_interaction_edge.T:
            # Compare each edge from the subset with all edges in the original
            match = (edge_index == edge.unsqueeze(1)).all(dim=0)

            if match.any():
                # If match found, append the index to the matching_indices list.
                matching_indices = torch.concat([matching_indices, match.nonzero(as_tuple=True)[0]])
            else:
                # If no match found, append the edge to the original and get its index
                edge_index = torch.cat([edge_index, edge.unsqueeze(1)], dim=1)
                edge_weight = torch.cat(
                    [edge_weight, torch.tensor([self.cutoff], device=edge_index.device, dtype=torch.long)]
                )
                last_index = torch.tensor([edge_index.size(1) - 1], device=edge_index.device, dtype=torch.long)
                matching_indices = torch.concat([matching_indices, last_index])
        return edge_index, edge_weight, matching_indices


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

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        mask_lp_edge_index: Tensor,
        lp_edge_attr: Tensor,
    ) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, mask_lp_edge_index, lp_edge_attr)
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

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
        mask_lp_edge_index: Tensor,
        lp_edge_attr: Tensor,
    ) -> Tensor:
        w_distance = self.compute_weight_edge_ligand(edge_weight, edge_attr)
        w_lp_interaction = torch.zeros_like(w_distance, device=w_distance.device)
        w_lp_interaction[mask_lp_edge_index, :] = self.nn_protein(lp_edge_attr)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, w_distance=w_distance, w_lp_interaction=w_lp_interaction)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, w_distance, w_lp_interaction) -> Tensor:
        return x_j * w_distance + x_j * w_lp_interaction
