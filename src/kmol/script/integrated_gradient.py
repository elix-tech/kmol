import pandas as pd
from pathlib import Path

from tqdm import tqdm

from mila.factories import AbstractScript

from ..core.config import Config
from ..core.helpers import SuperFactory
from ..data.streamers import GeneralStreamer
from ..data.loaders import AbstractLoader
from ..visualization.captum_network import AbstractCaptumNetwork

from ..core.logger import LOGGER as logging


class IntegratedGradientScript(AbstractScript):
    def __init__(self, config_path) -> None:
        super().__init__()
        self._config = Config.from_file(config_path)
        self.loader = SuperFactory.create(AbstractLoader, self._config.loader)
        streamer = GeneralStreamer(config=self._config)
        self.data_loader = iter(
            streamer.get(
                split_name="test",
                batch_size=1,
                shuffle=False,
                mode=GeneralStreamer.Mode.TEST,
            ).dataset
        )
        self.model = SuperFactory.create(AbstractCaptumNetwork, self._config.model)
        self.model.load_checkpoint(self._config.checkpoint_path)
        self.dataset = self.data_loader._dataset.dataset._dataset
        self.column_of_interest = (
            self.data_loader._dataset.dataset._input_columns
            + self.data_loader._dataset.dataset._target_columns
            + self.model.ig_outputs
        )

    def run(self):
        results = pd.DataFrame([], columns=self.column_of_interest)
        for data in tqdm(self.data_loader):
            try:
                ig = self.model.get_integrate_gradient(data.inputs)
            except Exception:
                logging.info("One element has been skipped")
            for i, col in enumerate(self.model.ig_outputs):
                self.dataset.loc[data.ids[0], col] = list(ig.values())[i]
            results = pd.concat([results, self.dataset.loc[data.ids, self.column_of_interest]])

        results.to_csv(Path(self._config.output_path) / "integrated_gradient_results.csv")
