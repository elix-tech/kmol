from pathlib import Path
import shutil
import time
from typing import Dict

from kmol.core.logger import LOGGER as logging

from ..factories import AbstractAggregator, AbstractScript
from ..services import IOManager


class ManualAggregationScript(AbstractScript, IOManager):
    def __init__(
        self, 
        chekpoint_paths: list, 
        aggregator_type: str, 
        aggregator_options: Dict = {}, 
        save_path: str = "data/logs/local/manual_aggregator.pt"
    ) -> None:

        self.chekpoint_paths = chekpoint_paths
        self.aggregator_type = aggregator_type
        self.aggregator_options = aggregator_options
        self.save_path = save_path
        if not Path(self.save_path).parent.exists():
            Path(self.save_path).parent.mkdir(parents=True)

    def aggregate(self) -> None:
        logging.info("Start local aggregation")

        aggregator: AbstractAggregator = self._reflect(self.aggregator_type)
        if self.aggregator_type == "mila.aggregators.WeightedTorchAggregator":
            self.create_tmp_weights()
        aggregator(**self.aggregator_options).run(checkpoint_paths=self.chekpoint_paths, save_path=self.save_path)

        logging.info("Aggregate model saved: [{}]".format(self.save_path))
    
    def create_tmp_weights(self) -> None:
        assert "weights" in self.aggregator_options.keys(), "WeightedTorchAggregator needs weights argument"
        weights = {}
        tmp_paths = []
        self.dir_name = f"kmol_manual_aggregator-{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}"
        if not Path(self.dir_name).exists():
            Path(self.dir_name).mkdir()
        for i, (checkpoint_path, weight) in enumerate(zip(self.chekpoint_paths, self.aggregator_options["weights"])):
            k = f"client_{i}"
            tmp_path = Path(self.dir_name) / f"{k}.{Path(checkpoint_path).name}"
            Path(tmp_path).symlink_to(Path(checkpoint_path).absolute())
            tmp_paths.append(str(tmp_path))
            weights.update({k: weight})
        
        self.chekpoint_paths = tmp_paths
        self.aggregator_options["weights"] = weights
            
    def clean_up(self) -> None:
        if hasattr(self, "dir_name"):
            shutil.rmtree(self.dir_name, ignore_errors=True)
    
    def run(self) -> None:
        # Box related stuffs maybe
        try:
            self.aggregate()
        finally:
            self.clean_up()
