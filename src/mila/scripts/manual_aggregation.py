
from pathlib import Path
from typing import Dict

from kmol.core.logger import LOGGER as logging

from ..factories import AbstractAggregator, AbstractScript
from ..services import IOManager


class ManualAggregationScript(AbstractScript, IOManager):
    def __init__(
        self, 
        chekpoints_paths: list, 
        aggregator_type: str, 
        aggregator_options: Dict = {}, 
        save_path: str = "data/logs/local/manual_aggregator.pt"
    ) -> None:

        self.chekpoints_paths = chekpoints_paths
        self.aggregator_type = aggregator_type
        self.aggregator_options = aggregator_options
        self.save_path = save_path
        if not Path(self.save_path).parent.exists():
            Path(self.save_path).parent.mkdir(parents=True)

    def aggregate(self) -> None:
        logging.info("Start local aggregation")

        aggregator: AbstractAggregator = self._reflect(self.aggregator_type)
        aggregator(**self.aggregator_options).run(checkpoint_paths=self.chekpoints_paths, save_path=self.save_path)

        logging.info("Aggregate model saved: [{}]".format(self.save_path))
    
    def run(self) -> None:
        # Box related stuffs maybe
        self.aggregate()
