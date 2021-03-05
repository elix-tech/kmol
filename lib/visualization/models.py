from collections import defaultdict
from copy import copy
from math import sqrt
from typing import Optional
from typing import Union, Dict, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx

from lib.data.resources import Data, Batch
from vendor.captum.attr import IntegratedGradients
from lib.visualization.sketchers import RdkitSketcher, GraphSketcher


class GNNExplainer(torch.nn.Module):
    """
    Description:
        This is the modified version of the PyTorch implementation of the
        GNN-Explainer model from the paper "GNNExplainer: Generating Explanations
        for Graph Neural Networks" (https://arxiv.org/abs/1903.03894) paper for
        identifying compact subgraph structures and small subsets node features
        that play a crucial role in a GNN’s node-predictions.

    Input Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train. (Default: 100)
        lr (float, optional): The learning rate to apply. (Default: 0.01)
        num_hops (int, optional): The number of hops the model is aggregating information from.
                                  If set to 'None', it will automatically try to detect this
                                  information based on the number of MessagePassing layers
                                  inside the model. (Default: None)

        log (bool, optional): If set to 'False', it will not log any learning progress. (Default: True)
    """

    def __init__(self, model, output_path: str, epochs: int = 100, lr: float = 0.01, num_hops: Optional[int] = None):
        super(GNNExplainer, self).__init__()

        self.coeffs = {
            "edge_size": 0.005,
            "edge_reduction": "sum",
            "node_feat_size": 1.0,
            "node_feat_reduction": "mean",
            "edge_ent": 1.0,
            "node_feat_ent": 0.1
        }

        self.EPS = 1e-15
        self.model = model
        self.epochs = epochs
        self.lr = lr

        self.sketcher = RdkitSketcher(output_path=output_path)
        self.__num_hops__ = num_hops

    def __set_masks__(self, x, edge_index):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1

        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow

        return "source_to_target"

    def __subgraph__(self, node_idx, x, edge_index, **kwargs) -> tuple:
        """
        In order to explain a node, get its k-hop computation graph.
        In order to explain the whole graph, get the whole computation graph.
        """

        num_nodes, num_edges = x.size(0), edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())

            x = x[subset]
        else:
            row, _ = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:] = True
            mapping = None

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]

            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __node_loss__(self, node_idx, log_logits, pred_label):
        """
        Calculate the loss for the network considering only a single node.
        The loss is the combination of the prediction loss and edge size loss.
        """

        loss = -log_logits[node_idx, pred_label[node_idx]]

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + self.EPS) - (1 - m) * torch.log(1 - m + self.EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.EPS) - (1 - m) * torch.log(1 - m + self.EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def __graph_loss__(self, log_logits, pred_label):
        """
        Calculate the loss for the network considering the whole graph.
        The loss is the combination of the prediction loss and edge size loss.
        """

        loss = -torch.log(log_logits[0, pred_label])

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs["edge_size"] * m.sum()
        ent = -m * torch.log(m + self.EPS) - (1 - m) * torch.log(1 - m + self.EPS)
        loss = loss + self.coeffs["edge_ent"] * ent.mean()

        return loss

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        """
        Description:
            Learns and returns a node feature mask and an edge mask that play a crucial
            role to explain the prediction made by the GNN for node 'node_idx'.
        Input Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        Output Args:
            edge_mask (Tensor): The edge mask.
            node_feat_mask (Tensor): The node feature mask.
        """

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(node_idx, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            log_logits = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__node_loss__(mapping, log_logits, pred_label)
            loss.backward()
            optimizer.step()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return edge_mask, node_feat_mask

    def explain_graph(self, x, edge_index, target: int, **kwargs):
        """
        Description:
            Learns and returns an edge mask that play a crucial role to explain
            the prediction made by the GNN for the whole graph.
        Input Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        Output Args:
            edge_mask (Tensor): The edge mask.
        """

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)
        x, edge_index, _, _, kwargs = self.__subgraph__(None, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = logits[:, target]

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)
        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()

            logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred = torch.softmax(logits, 1)
            loss = self.__graph_loss__(pred, pred_label)

            loss.sum().backward()
            optimizer.step()

        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask = edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None, threshold=None, **kwargs):
        """
        Description:
            Visualizes the subgraph around 'node_idx' or the whole graph given an edge mask 'edge_mask'.
        Input Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used as node colorings. (Default: 'None')
            threshold (float, optional): Sets a threshold for visualizing important edges. If set to 'None', it will
                                         visualize all edges with transparancy indicating the importance of edges.
                                         (Default: 'None')
            **kwargs (optional): Additional arguments passed to 'nx.draw'.
        Output Args:
            ax (matplotlib.axes.Axes): Matplotlib axes with the plotted graph.
            g (networkx.DiGraph): The generated digraph.
        """

        if edge_mask.size(0) != edge_index.size(1):
            raise ValueError("Edge mask does not match edge index")

        if node_idx is not None:
            # Only operate on a k-hop subgraph around 'node_idx'.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]

            if threshold is not None:
                edge_mask = (edge_mask >= threshold).to(torch.float)

            plt.figure(figsize=(10, 5))

        else:
            if threshold is not None:
                edge_mask = (edge_mask >= threshold).to(torch.float)

            subset = []

            for index, _ in enumerate(edge_mask):
                node_a = edge_index[0, index]
                node_b = edge_index[1, index]

                if node_a not in subset:
                    subset.append(node_a.cpu().item())

                if node_b not in subset:
                    subset.append(node_b.cpu().item())

            edge_list = []

            for index, edge in enumerate(edge_mask):
                if edge:
                    edge_list.append((edge_index[0, index].cpu(), edge_index[1, index].cpu()))

            plt.figure(figsize=(20, 10))

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index, att=edge_mask, y=y, num_nodes=y.size(0)).to("cpu")
        G = to_networkx(data, node_attrs=["y"], edge_attrs=["att"])

        if type(subset) == list:
            mapping = {k: i for k, i in enumerate(subset)}
        else:
            mapping = {k: i for k, i in enumerate(subset.tolist())}

        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs["node_size"] = kwargs.get("node_size") or 800
        node_kwargs["cmap"] = "cool"

        label_kwargs = copy(kwargs)
        label_kwargs["font_size"] = kwargs.get("font_size") or 10

        pos = nx.spring_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate("", xy=pos[target], xycoords="data", xytext=pos[source], textcoords="data", arrowprops={
                "arrowstyle": "->", "alpha": max(data["att"], 0.1), "shrinkA": sqrt(node_kwargs["node_size"]) / 2.0,
                "shrinkB": sqrt(node_kwargs["node_size"]) / 2.0, "connectionstyle": "arc3,rad=0.1"
            })

        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        plt.axis("off")
        return ax, G

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def visualize(self, data: Batch, target: int, save_path: str) -> None:
        edge_mask = self.explain_graph(data.inputs["graph"].x, data.inputs["graph"].edge_index, target)
        self.sketcher.draw(data, save_path, edge_mask)


class IntegratedGradientsExplainer:
    """
    Implements the Integrated Gradients explainability technique introduced in https://arxiv.org/abs/1703.01365.
    The core implementation relies on captum: https://github.com/pytorch/captum
    """

    def __init__(self, model: torch.nn.Module, output_path: str):
        self.model = model
        self.model.eval()

        self.sketcher = GraphSketcher(output_path=output_path)

    def explain(self, data: Union[Data, Batch], target: int) -> Dict[int, np.ndarray]:
        graphs = data.inputs["graph"]

        number_nodes = graphs.x.shape[0]
        input_mask = graphs.x.requires_grad_(True).to(graphs.x.device)
        integrated_gradients = IntegratedGradients(self.model_forward)

        mask = integrated_gradients.attribute(
            input_mask, target=target, additional_forward_args=(data,), internal_batch_size=number_nodes
        )

        node_mask = np.abs(mask.cpu().detach().numpy()).sum(axis=1)
        return self.per_mol_mask(data, node_mask)

    def per_mol_mask(self, data: Data, node_masks: np.ndarray) -> Dict[int, np.ndarray]:
        node_mask_per_mol = defaultdict(list)
        batch_ids = data.inputs["graph"].batch

        for node_mask_value, batch_id in zip(node_masks, batch_ids):
            node_mask_per_mol[batch_id.item()].append(node_mask_value)

        for batch_id, node_mask in node_mask_per_mol.items():
            if np.max(node_mask) > 0:
                node_mask_per_mol[batch_id] = (np.array(node_mask) / np.array(node_mask).max())

        return node_mask_per_mol

    def model_forward(self, node_mask: torch.Tensor, data: Union[Data, Batch]) -> torch.Tensor:
        data.inputs["graph"].x = node_mask

        if not hasattr(data.inputs["graph"], "batch"):
            data.inputs["graph"].batch = torch.zeros(
                data.inputs["graph"].x.shape[0], dtype=int
            ).to(data.inputs["graph"].x.device)

        return self.model(data.inputs)

    def to_molecule(self, data: torch_geometric.data.Data) -> nx.Graph:
        mol = Chem.MolFromSmiles(data.smiles)
        g = to_networkx(data, node_attrs=["x"])

        for (u, data), atom in zip(g.nodes(data=True), mol.GetAtoms()):
            data["name"] = atom.GetSymbol()
            del data["x"]

        return g

    def to_mol_list(self, data: Data) -> Tuple[List, List]:
        if isinstance(data.inputs["graph"], torch_geometric.data.Batch):
            data_list = data.inputs["graph"].to_data_list()
            dataset_sample_ids = data.ids
        else:
            data_list = [data.inputs["graph"]]
            dataset_sample_ids = [data.id_]

        return data_list, dataset_sample_ids

    def visualize(self, data: Union[Data, Batch], target: int, save_path: str) -> None:
        node_mask_per_mol = self.explain(data, target)
        data_list, dataset_sample_ids = self.to_mol_list(data)

        for (batch_sample_id, node_mask), dataset_sample_id in zip(node_mask_per_mol.items(), dataset_sample_ids):
            data = data_list[batch_sample_id]
            mol = self.to_molecule(data)

            plt.figure(figsize=(10, 5))
            plt.title(f"Integrated Gradients {data.smiles}")

            self.sketcher.draw(graph=mol, save_path=save_path, node_mask=node_mask)
