import logging
from abc import ABCMeta

import numpy as np
import torch
from torch.nn.modules.loss import _Loss as AbstractCriterion
from torch.optim import Optimizer as AbstractOptimizer
from torch.optim.lr_scheduler import _LRScheduler as AbstractLearningRateScheduler
from torch.utils.data import DataLoader as TorchDataLoader

from lib.core.config import Config
from lib.core.exceptions import CheckpointNotFound
from lib.core.helpers import Timer, SuperFactory, Namespace
from lib.data.resources import Batch
from lib.model.architectures import AbstractNetwork
from lib.model.metrics import PredictionProcessor
from lib.model.trackers import ExponentialAverageMeter
from lib.core.observers import EventManager


class AbstractExecutor(metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config
        self._timer = Timer()
        self._start_epoch = 0

        self._network = None
        self._setup_network()

        self._optimizer = None
        self._criterion = None
        self._scheduler = None

    def _load_checkpoint(self) -> None:
        if self._config.checkpoint_path is None:
            raise CheckpointNotFound

        logging.info("Restoring from Checkpoint: {}".format(self._config.checkpoint_path))
        info = torch.load(self._config.checkpoint_path, map_location=self._config.get_device())

        self._network.load_state_dict(info["model"])

        if self._optimizer and "optimizer" in info:
            self._optimizer.load_state_dict(info["optimizer"])

        if self._scheduler and "scheduler" in info:
            self._scheduler.load_state_dict(info["scheduler"])

        if "epoch" in info:
            self._start_epoch = info["epoch"]

    def _setup_network(self) -> None:
        network = SuperFactory.create(AbstractNetwork, self._config.model)

        payload = Namespace(network=network, config=self._config)
        EventManager.dispatch_event(event_name="after_network_create", payload=payload)

        self._network = payload.network
        if self._config.should_parallelize():
            self._network = torch.nn.DataParallel(self._network, device_ids=self._config.enabled_gpus)

        self._network.to(self._config.get_device())


class Trainer(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)
        self._loss_tracker = ExponentialAverageMeter(smoothing_factor=0.95)

        self._metric_trackers = {
            name: ExponentialAverageMeter(smoothing_factor=0.9) for name in self._config.train_metrics
        }
        self._metric_computer = PredictionProcessor(
            metrics=self._config.train_metrics, threshold=self._config.threshold, error_value=0
        )

    def _setup(self, training_examples: int) -> None:
        self._criterion = SuperFactory.create(
            AbstractCriterion, self._config.criterion
        ).to(self._config.get_device())

        self._optimizer = SuperFactory.create(AbstractOptimizer, self._config.optimizer, {
            "params": self._network.parameters()
        })

        self._scheduler = SuperFactory.create(AbstractLearningRateScheduler, self._config.scheduler, {
            "optimizer": self._optimizer,
            "steps_per_epoch": max(training_examples // self._config.batch_size, 1)
        })

        try:
            self._load_checkpoint()
        except CheckpointNotFound:
            pass

        self._network = self._network.train()
        logging.debug(self._network)

    def run(self, data_loader: TorchDataLoader):

        dataset_size = len(data_loader.dataset)
        self._setup(training_examples=dataset_size)

        payload = Namespace(trainer=self, data_loader=data_loader)
        EventManager.dispatch_event(event_name="before_train_start", payload=payload)

        for epoch in range(self._start_epoch + 1, self._config.epochs + 1):

            for iteration, data in enumerate(data_loader, start=1):
                self._optimizer.zero_grad()
                outputs = self._network(data.inputs)

                payload = Namespace(input=outputs, target=data.outputs)
                EventManager.dispatch_event(event_name="before_criterion", payload=payload)
                loss = self._criterion(**vars(payload))

                loss.backward()
                self._optimizer.step()

                self._update_trackers(loss.item(), data.outputs, outputs)
                if iteration % self._config.log_frequency == 0:
                    self.log(epoch, iteration, dataset_size)

            self._scheduler.step()
            self._reset_trackers()

            self.save(epoch)

    def _update_trackers(self, loss: float, ground_truth: torch.Tensor, logits: torch.Tensor) -> None:
        self._loss_tracker.update(loss)

        metrics = self._metric_computer.compute_metrics([ground_truth], [logits])
        averages = self._metric_computer.compute_statistics(metrics, (np.mean,))

        for metric_name, tracker in self._metric_trackers.items():
            tracker.update(getattr(averages, metric_name)[0])

    def _reset_trackers(self) -> None:
        self._loss_tracker.reset()

        for tracker in self._metric_trackers.values():
            tracker.reset()

    def save(self, epoch: int) -> None:
        info = {
            "epoch": epoch,
            "model": self._network.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict()
        }

        model_path = "{}checkpoint.{}".format(self._config.output_path, epoch)
        logging.info("Saving checkpoint: {}".format(model_path))

        torch.save(info, model_path)

    def log(self, epoch: int, iteration: int, dataset_size: int) -> None:
        processed_samples = iteration * self._config.batch_size
        message = "epoch: {} - iteration: {} - examples: {} - loss: {:.4f} - time elapsed: {} - progress: {}".format(
            epoch,
            iteration,
            processed_samples,
            self._loss_tracker.get(),
            str(self._timer),
            round(processed_samples / dataset_size, 4)
        )

        for name, tracker in self._metric_trackers.items():
            message += " - {}: {:.4f}".format(name, tracker.get())

        payload = Namespace(message=message, epoch=epoch, iteration=iteration, dataset_size=dataset_size, trainer=self)
        EventManager.dispatch_event(event_name="before_train_progress_log", payload=payload)

        logging.info(payload.message)


class Predictor(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

        self._load_checkpoint()
        self._network = self._network.eval()

    @torch.no_grad()
    def run(self, batch: Batch) -> torch.Tensor:
        return self._network(batch.inputs)


class Evaluator(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

        self._predictor = Predictor(config=self._config)
        self._processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)

    def run(self, data_loader: TorchDataLoader) -> Namespace:

        ground_truth = []
        logits = []

        for batch in data_loader:
            ground_truth.append(batch.outputs)
            logits.append(self._predictor.run(batch))

        return self._processor.compute_metrics(ground_truth=ground_truth, logits=logits)
