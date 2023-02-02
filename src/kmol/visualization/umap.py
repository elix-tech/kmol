import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from ..core.logger import LOGGER as logging


class UMAPVisualizer:
    def __init__(self, output_path, **kwargs):
        kwargs.pop("label_index", None)
        self.output_path = output_path
        self.reducer = umap.UMAP(**kwargs)

    def visualize(self, features: np.ndarray, labels: Optional[np.ndarray] = None, label_name: Optional[str] = None):
        plt.rcParams["savefig.dpi"] = 300

        embeddings = self.reducer.fit_transform(features)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], s=1, c=labels)
        if labels is not None:
            cbar = plt.colorbar()
            cbar.set_label(label_name)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP Projection")
        save_path = Path(self.output_path) / "umap.png"
        plt.savefig(save_path)
        logging.info(f"UMAP saved in: {str(save_path)}")
