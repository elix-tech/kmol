import logging
import math
from abc import ABCMeta
from copy import copy
from functools import partial
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.modules.loss import _Loss as AbstractCriterion
from torch.optim import Optimizer as AbstractOptimizer
from torch.optim.lr_scheduler import _LRScheduler as AbstractLearningRateScheduler, ExponentialLR
from tqdm import tqdm

from lib.core.config import Config
from lib.core.exceptions import CheckpointNotFound
from lib.core.helpers import Timer, SuperFactory, Namespace
from lib.core.observers import EventManager
from lib.data.resources import Batch, LoadedContent
from lib.model.architectures import AbstractNetwork
from lib.model.metrics import PredictionProcessor
from lib.model.trackers import ExponentialAverageMeter


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

        payload = Namespace(executor=self, info=info)
        EventManager.dispatch_event(event_name="before_checkpoint_load", payload=payload)

        self._network.load_state_dict(info["model"])

        if not self._config.is_finetuning:
            if self._optimizer and "optimizer" in info:
                self._optimizer.load_state_dict(info["optimizer"])

            if self._scheduler and "scheduler" in info:
                self._scheduler.load_state_dict(info["scheduler"])

            if "epoch" in info:
                self._start_epoch = info["epoch"]

        payload = Namespace(executor=self)
        EventManager.dispatch_event(event_name="after_checkpoint_load", payload=payload)

    def _setup_network(self) -> None:
        self._network = SuperFactory.create(AbstractNetwork, self._config.model)

        payload = Namespace(executor=self, config=self._config)
        EventManager.dispatch_event(event_name="after_network_create", payload=payload)

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

        self._scheduler = self._initialize_scheduler(optimizer=self._optimizer, training_examples=training_examples)

        try:
            self._load_checkpoint()
        except CheckpointNotFound:
            pass

        self._network = self._network.train()
        logging.debug(self._network)

    def _initialize_scheduler(
            self, optimizer: AbstractOptimizer, training_examples: int
    ) -> AbstractLearningRateScheduler:

        return SuperFactory.create(AbstractLearningRateScheduler, self._config.scheduler, {
            "optimizer": optimizer,
            "steps_per_epoch": math.ceil(training_examples / self._config.batch_size)
        })

    def run(self, data_loader: LoadedContent):

        self._setup(training_examples=data_loader.samples)

        payload = Namespace(trainer=self, data_loader=data_loader)
        EventManager.dispatch_event(event_name="before_train_start", payload=payload)

        for epoch in range(self._start_epoch + 1, self._config.epochs + 1):

            for iteration, data in enumerate(data_loader.dataset, start=1):
                self._optimizer.zero_grad()
                outputs = self._network(data.inputs)

                payload = Namespace(features=data, logits=outputs)
                EventManager.dispatch_event(event_name="before_criterion", payload=payload)

                loss = self._criterion(payload.logits, payload.features.outputs)
                loss.backward()

                self._optimizer.step()
                if self._config.is_stepwise_scheduler:
                    self._scheduler.step()

                self._update_trackers(loss.item(), data.outputs, outputs)
                if iteration % self._config.log_frequency == 0:
                    self.log(epoch, iteration, data_loader.samples)

            if not self._config.is_stepwise_scheduler:
                self._scheduler.step()

            self._reset_trackers()
            self.save(epoch)

        EventManager.dispatch_event(event_name="after_train_end", payload=payload)

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

        payload = Namespace(info=info)
        EventManager.dispatch_event(event_name="before_checkpoint_save", payload=payload)

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

    def run(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            logits = self._network(batch.inputs)

            payload = Namespace(features=batch, logits=logits)
            EventManager.dispatch_event("after_predict", payload=payload)

            return payload.logits

    def run_all(self, data_loader: LoadedContent) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ground_truth = []
        logits = []

        for batch in data_loader.dataset:
            ground_truth.append(batch.outputs)
            logits.append(self.run(batch))

        return ground_truth, logits


class Evaluator(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

        self._predictor = Predictor(config=self._config)
        self._processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)

    def run(self, data_loader: LoadedContent) -> Namespace:
        ground_truth, logits = self._predictor.run_all(data_loader=data_loader)
        return self._processor.compute_metrics(ground_truth=ground_truth, logits=logits)


class Pipeliner(AbstractExecutor):

    def __init__(self, config: Config):
        self._config = config

        self._trainer = Trainer(self._config)
        self._processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)
        self._predictor = None

    def initialize_predictor(self) -> "Pipeliner":
        self._predictor = Predictor(config=self._config)
        return self

    def train(self, data_loader: LoadedContent) -> None:
        self._trainer.run(data_loader=data_loader)

    def evaluate(self, data_loader: LoadedContent) -> Namespace:
        ground_truth, logits = self.predict(data_loader=data_loader)
        return self._processor.compute_metrics(ground_truth=ground_truth, logits=logits)

    def predict(self, data_loader: LoadedContent) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._predictor.run_all(data_loader=data_loader)

    def evaluate_all(self, data_loader: LoadedContent) -> List[Namespace]:
        results = []

        for checkpoint_path in self.find_all_checkpoints():
            config = copy(self._config)
            config.checkpoint_path = checkpoint_path

            evaluator = Evaluator(config=config)
            results.append(evaluator.run(data_loader=data_loader))

        return results

    def find_all_checkpoints(self) -> List[str]:
        checkpoint_paths = glob(self._config.output_path + "*")
        checkpoint_paths = sorted(checkpoint_paths)
        return sorted(checkpoint_paths, key=len)

    def find_best_checkpoint(self, data_loader: LoadedContent) -> "Pipeliner":
        results = self.evaluate_all(data_loader=data_loader)

        per_target_best = Namespace.reduce(results, partial(np.argmax, axis=0))
        per_target_best = getattr(per_target_best, self._config.target_metric)

        self._config.checkpoint_path = "{}checkpoint.{}".format(
            self._config.output_path, np.argmax(np.bincount(per_target_best)) + 1
        )

        self.initialize_predictor()
        return self


class ThresholdFinder(Evaluator):

    def run(self, data_loader: LoadedContent) -> List[float]:
        ground_truth, logits = self._predictor.run_all(data_loader=data_loader)
        return self._processor.find_best_threshold(ground_truth=ground_truth, logits=logits)


class LearningRareFinder(Trainer):
    """
    Runs training for a given number of steps to find appropriate lr value.
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    MAXIMUM_LEARNING_RATE = 0.1
    MINIMUM_LEARNING_RATE = 1e-5

    def _initialize_scheduler(
            self, optimizer: AbstractOptimizer, training_examples: int
    ) -> AbstractLearningRateScheduler:

        gamma = max(training_examples // self._config.batch_size, 1)
        gamma = np.log(self.MAXIMUM_LEARNING_RATE / self.MINIMUM_LEARNING_RATE) / gamma
        gamma = float(np.exp(gamma))

        return ExponentialLR(optimizer=optimizer, gamma=gamma)

    def run(self, data_loader: LoadedContent) -> None:

        self._setup(training_examples=data_loader.samples)

        payload = Namespace(trainer=self, data_loader=data_loader)
        EventManager.dispatch_event(event_name="before_train_start", payload=payload)

        learning_rate_records = []
        loss_records = []

        try:
            with tqdm(total=data_loader.batches) as progress_bar:
                for iteration, data in enumerate(data_loader.dataset, start=1):
                    self._optimizer.zero_grad()
                    outputs = self._network(data.inputs)

                    payload = Namespace(input=outputs, target=data.outputs)
                    EventManager.dispatch_event(event_name="before_criterion", payload=payload)
                    loss = self._criterion(**vars(payload))

                    loss.backward()
                    self._optimizer.step()

                    self._scheduler.step()
                    self._loss_tracker.update(loss.item())

                    learning_rate_records.append(self._get_learning_rate())
                    loss_records.append(self._loss_tracker.get())

                    progress_bar.update(1)
        except (KeyboardInterrupt, RuntimeError):
            pass

        self._plot(learning_rate_records, loss_records)

    def _get_learning_rate(self) -> float:
        return self._optimizer.param_groups[0]["lr"]

    def _plot(self, learning_rate_records: List[float], loss_records: List[float]) -> None:
        import matplotlib.pyplot as plt

        plt.plot(learning_rate_records, loss_records)
        plt.xscale("log")

        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")

        plt.show()
