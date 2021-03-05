import io
import os
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from cairosvg import svg2png
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from lib.data.resources import Batch


class AbstractSketcher(metaclass=ABCMeta):

    def __init__(self, output_path: str):
        self._output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    @abstractmethod
    def draw(self, *args, **kwargs) -> None:
        raise NotImplementedError


class RdkitSketcher(AbstractSketcher):

    def _generate_color_code(
            self, highlight_intensities: List[float], color_switch_threshold: float = 0.3
    ) -> Dict[int, tuple]:

        if highlight_intensities is None:
            return {}

        colors = {}
        for highlight_ind, highlight_intensity in enumerate(highlight_intensities):
            if highlight_intensity >= color_switch_threshold:
                colors.update({
                    highlight_ind: (1, round(1.1 - highlight_intensity, 3), round(1.1 - highlight_intensity, 3))
                })
            else:
                colors.update({
                    highlight_ind: (round(highlight_intensity, 3) + 0.4, round(highlight_intensity, 3) + 0.4, 1)
                })

        return colors

    def _draw(
            self, smiles: str, save_path: str, x: int = 300, y: int = 200,
            highlight_atoms: List[float] = None, highlight_bonds: List[float] = None
    ) -> Image:
        mol = AllChem.MolFromSmiles(smiles)
        AllChem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)

        highlight_atoms = self._generate_color_code(highlight_atoms)
        highlight_bonds = self._generate_color_code(highlight_bonds)

        try:
            mol.GetAtomWithIdx(0).GetExplicitValence()
        except RuntimeError:
            mol.UpdatePropertyCache(False)

        try:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=True)
        except ValueError:
            mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)

        drawer = rdMolDraw2D.MolDraw2DSVG(x, y)
        drawer.DrawMolecule(
            mol, highlightAtoms=list(highlight_atoms.keys()), highlightAtomColors=highlight_atoms,
            highlightBonds=list(highlight_bonds.keys()), highlightBondColors=highlight_bonds
        )

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")

        Image.open(io.BytesIO(svg2png(svg))).save(save_path, "png")

    def draw(self, batch: Batch, save_path: str, mask: torch.Tensor) -> None:
        smiles = batch.inputs["graph"].smiles[0]
        source_edge_index = batch.inputs["graph"].edge_index[0]
        target_edge_index = batch.inputs["graph"].edge_index[1]

        highlights = []
        for bond in Chem.MolFromSmiles(smiles).GetBonds():
            value = []

            for i in range(len(source_edge_index)):
                if (
                        bond.GetBeginAtomIdx() == source_edge_index[i] and bond.GetEndAtomIdx() == target_edge_index[i]
                ) or (
                        bond.GetBeginAtomIdx() == target_edge_index[i] and bond.GetEndAtomIdx() == source_edge_index[i]
                ):
                    value.append(mask[i])

            highlights.append(float(np.mean(value)))

        save_path = "{}/{}".format(self._output_path, save_path)
        self._draw(smiles=smiles, save_path=save_path, x=1000, y=800, highlight_bonds=highlights)


class GraphSketcher(AbstractSketcher):

    def draw(self, graph: nx.Graph, save_path: str, node_mask: torch.Tensor = None) -> None:
        graph = graph.copy().to_undirected()
        node_labels = {}

        for u, data in graph.nodes(data=True):
            node_labels[u] = data["name"]

        pos = nx.planar_layout(graph)
        pos = nx.spring_layout(graph, pos=pos)

        node_color = "azure" if node_mask is None else node_mask
        nx.draw(graph, pos=pos, labels=node_labels, edge_color="black", cmap=plt.cm.Blues, node_color=node_color)

        plt.savefig("{}/{}".format(self._output_path, save_path), dpi=120)
        plt.close()
