from pathlib import Path
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss as AbstractCriterion
from torch.optim import Optimizer as AbstractOptimizer

from mila.factories import AbstractScript
from kmol.core.config import Config
from kmol.data.streamers import GeneralStreamer
from kmol.core.helpers import SuperFactory
from kmol.model.architectures import AbstractNetwork


class CustomTrainIter(TrainDataLoaderIter, ValDataLoaderIter):
    """
    Kmol code
    """

    def inputs_labels_from_batch(self, batch_data):
        return batch_data.inputs, batch_data.outputs


class LrFinderScript(AbstractScript):
    """
    https://github.com/davidtvs/pytorch-lr-finder
    """

    def __init__(self, cfg_path, path_to_save, type_process="fast") -> None:
        assert type_process in ["fast", "accurate"], f"type_process must be either fast or accurate got '{type_process}'"
        self._config = Config.from_file(cfg_path, job_command="eval")
        self.path_to_save = path_to_save
        self.type_process = type_process
        self.generate_training_materials()

    def generate_training_materials(self):
        streamer = GeneralStreamer(config=self._config)
        assert self._config.validation_split in streamer.splits, "Need a validation split to launch r-finder"
        self.train_loader = streamer.get(
            split_name=self._config.train_split,
            batch_size=self._config.batch_size,
            shuffle=True,
            mode=GeneralStreamer.Mode.TRAIN,
        )
        self.val_loader = streamer.get(
            split_name=self._config.validation_split,
            batch_size=self._config.batch_size,
            shuffle=False,
            mode=GeneralStreamer.Mode.TEST,
        )

        self.train_loader = CustomTrainIter(self.train_loader.dataset)
        self.val_loader = CustomTrainIter(self.val_loader.dataset)

        self.model = SuperFactory.create(AbstractNetwork, self._config.model)

        self.criterion = SuperFactory.create(AbstractCriterion, self._config.criterion)

        self.optimizer = SuperFactory.create(
            AbstractOptimizer,
            self._config.optimizer,
            {"params": self.model.parameters()},
        )

    def fastai_lr_finder(self):
        self.lr_finder.range_test(self.train_loader, start_lr=1.0e-5, end_lr=100, num_iter=1000)

    def accurate_lr_finder(self):
        self.lr_finder.range_test(
            self.train_loader, val_loader=self.val_loader, start_lr=1.0e-5, end_lr=1, num_iter=1000, step_mode="linear"
        )

    def save_plot(self):
        self.lr_finder.plot(log_lr=self.type_process == "fast")
        Path(self.path_to_save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.path_to_save)
        plt.close()

    def run(self):
        self.lr_finder = LRFinder(self.model, self.optimizer, self.criterion, self._config.get_device())
        if self.type_process == "accurate":
            self.accurate_lr_finder()
        elif self.type_process == "fast":
            self.fastai_lr_finder()
        self.save_plot()
