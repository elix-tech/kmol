import math
from abc import ABCMeta
from copy import copy
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.modules.loss import _Loss as AbstractCriterion
from torch.optim import Optimizer as AbstractOptimizer
from torch.optim.lr_scheduler import _LRScheduler as AbstractLearningRateScheduler, ExponentialLR
from tqdm import tqdm

from .architectures import AbstractNetwork, EnsembleNetwork
from .metrics import PredictionProcessor
from .trackers import ExponentialAverageMeter
from ..core.config import Config
from ..core.exceptions import CheckpointNotFound
from ..core.logger import LOGGER as logging
from ..core.helpers import Timer, SuperFactory, Namespace, HookProbe
from ..core.observers import EventManager
from ..data.resources import Batch, LoadedContent


class AbstractExecutor(metaclass=ABCMeta):
    def __init__(self, config: Config):
        self.config = config
        self._timer = Timer()
        self._start_epoch = 0
        self._device = self.config.get_device()

        self.network = None
        self._setup_network()

        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def _to_device(self, batch):
        batch.outputs = batch.outputs.to(self._device)
        for key, values in batch.inputs.items():
            try:
                batch.inputs[key] = values.to(self._device)
            except (AttributeError, ValueError):
                pass

    def _load_checkpoint(self, train: bool = False) -> None:
        self.network.load_checkpoint(self.config.checkpoint_path, self.config.get_device())

        if not self.config.is_finetuning and train:
            info = torch.load(self.config.checkpoint_path, map_location=self.config.get_device())
            if self.optimizer and "optimizer" in info:
                self.optimizer.load_state_dict(info["optimizer"])

            if self.scheduler and "scheduler" in info:
                self.scheduler.load_state_dict(info["scheduler"])

            if "epoch" in info:
                self._start_epoch = info["epoch"]

        payload = Namespace(executor=self)
        EventManager.dispatch_event(event_name="after_checkpoint_load", payload=payload)

    def _setup_network(self) -> None:
        self.network = SuperFactory.create(AbstractNetwork, self.config.model)

        payload = Namespace(executor=self, config=self.config)
        EventManager.dispatch_event(event_name="after_network_create", payload=payload)

        if self.config.should_parallelize():
            self.network = torch.nn.DataParallel(self.network, device_ids=self.config.enabled_gpus)

        self.network.to(self.config.get_device())


class Trainer(AbstractExecutor):
    def __init__(self, config: Config):
        super().__init__(config)
        self._loss_tracker = ExponentialAverageMeter(smoothing_factor=0.95)

        self._metric_trackers = {
            name: ExponentialAverageMeter(smoothing_factor=0.9) for name in self.config.train_metrics
        }
        self._metric_computer = PredictionProcessor(
            metrics=self.config.train_metrics,
            threshold=self.config.threshold,
        )

    def _setup(self, training_examples: int) -> None:
        self.criterion = SuperFactory.create(AbstractCriterion, self.config.criterion).to(self.config.get_device())

        self.optimizer = SuperFactory.create(
            AbstractOptimizer,
            self.config.optimizer,
            {"params": self.network.parameters()},
        )

        self.scheduler = self._initialize_scheduler(optimizer=self.optimizer, training_examples=training_examples)

        try:
            self._load_checkpoint(train=True)
        except CheckpointNotFound:
            pass

        self.network = self.network.train()
        logging.debug(self.network)

    def _initialize_scheduler(
            self, optimizer: AbstractOptimizer, training_examples: int
    ) -> AbstractLearningRateScheduler:

        return SuperFactory.create(
            AbstractLearningRateScheduler,
            self.config.scheduler,
            {
                "optimizer": optimizer,
                "steps_per_epoch": math.ceil(training_examples / self.config.batch_size),
            },
        )

    def run(self, data_loader: LoadedContent, val_loader: Optional[LoadedContent] = None):

        self._setup(training_examples=data_loader.samples)

        initial_payload = Namespace(trainer=self, data_loader=data_loader)
        EventManager.dispatch_event(event_name="before_train_start", payload=initial_payload)
        best_metric = -np.inf
        for epoch in range(self._start_epoch + 1, self.config.epochs + 1):
            self._train_epoch(data_loader, epoch)
            val_metrics = self._validation(val_loader)
            best_metric, new_best = self._check_best(epoch, val_metrics, best_metric)

            self.log(epoch, val_metrics, new_best)
            self._reset_trackers()

        EventManager.dispatch_event(event_name="after_train_end", payload=initial_payload)

    def _training_step(self, batch):
        self._to_device(batch)
        self.optimizer.zero_grad()
        outputs = self.network(batch.inputs)
        payload = Namespace(features=batch, logits=outputs, extras=[])
        EventManager.dispatch_event(event_name="before_criterion", payload=payload)

        loss = self.criterion(payload.logits, payload.features.outputs, *payload.extras)
        loss.backward()

        self.optimizer.step()
        if self.config.is_stepwise_scheduler:
            self.scheduler.step()

        self._update_trackers(loss.item(), batch.outputs, outputs)

    def _train_epoch(self, train_loader, epoch):
        self.network.train()
        pbar = tqdm(train_loader.dataset, total=len(train_loader.dataset), leave=False)
        iteration = 1
        for batch in pbar:
            self._training_step(batch)
            if iteration % self.config.log_frequency == 0:
                pbar.set_description(f"Epoch {epoch} | Train Loss: {self._loss_tracker.get():.5f}")
            iteration += 1
        if not self.config.is_stepwise_scheduler:
            self.scheduler.step()
        logging.only_log_file(str(pbar))

    @torch.no_grad()
    def _validation(self, val_loader):
        if val_loader is None:
            return Namespace()
        ground_truth = []
        logits = []
        self.network.eval()
        for batch in tqdm(val_loader.dataset, leave=False):
            self._to_device(batch)
            ground_truth.append(batch.outputs)
            logits.append(self.network(batch.inputs))

        metrics = self._metric_computer.compute_metrics(ground_truth, logits)
        averages = self._metric_computer.compute_statistics(metrics, (np.mean,))

        return averages

    def _check_best(self, epoch, val_metrics, best_metric):
        if val_metrics == Namespace():
            new_best = True
            target_metric = best_metric
        else:
            target_metric = getattr(val_metrics, self.config.target_metric)[0]
            new_best = target_metric > best_metric
        if new_best:
            best_metric = target_metric
            self.save(epoch)
        return best_metric, new_best


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
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        suffix = "best" if self.config.overwrite_checkpoint else epoch
        model_path = Path(self.config.output_path) / f"checkpoint_{suffix}.pt"
        logging.info("Saving checkpoint: {}".format(model_path))

        payload = Namespace(info=info)
        EventManager.dispatch_event(event_name="before_checkpoint_save", payload=payload)

        torch.save(info, model_path)

    def log(self, epoch: int, val_metrics: Namespace, new_best: bool) -> None:
        message = "epoch: {} - Train loss: {:.4f} - time elapsed: {}".format(
            epoch,
            self._loss_tracker.get(),
            str(self._timer),
        )

        for name, tracker in self._metric_trackers.items():
            message += " - Train {}: {:.4f}".format(name, tracker.get())

        for name, value in vars(val_metrics).items():
            message += " - Val {}: {:.4f}".format(name, value[0])

        message += " (New best)" if new_best else ""

        payload = Namespace(
            message=message,
            epoch=epoch,
            trainer=self,
        )
        EventManager.dispatch_event(event_name="before_train_progress_log", payload=payload)

        logging.info(payload.message)

        with (Path(self.config.output_path) / "logs.txt").open("a") as f:
            f.write(message + "\n")

class Predictor(AbstractExecutor):
    def __init__(self, config: Config):
        super().__init__(config)

        self._load_checkpoint()
        self.network = self.network.eval()
        self.probe = None

    def set_hook_probe(self):
        if isinstance(self.network, EnsembleNetwork):
            raise ValueError("Probing hidden layers is not defined for Ensembles."
                             " Please change 'probe_layer' parameter to 'null' or use a different type of network.")
        else:
            self.probe = HookProbe(self.network, self.config.probe_layer)

    def run(self, batch: Batch) -> torch.Tensor:
        self._to_device(batch)
        with torch.no_grad():
            if self.config.probe_layer is not None:
                self.set_hook_probe()

            if self.config.inference_mode == "mc_dropout":
                outputs = self.network.mc_dropout(
                    batch.inputs,
                    dropout_prob=self.config.mc_dropout_probability,
                    n_iter=self.config.mc_dropout_iterations,
                )
            else:
                outputs = self.network(batch.inputs)

            if isinstance(outputs, torch.Tensor):
                outputs = {"logits": outputs}

            if self.probe is not None:
                outputs["hidden_layer"] = self.probe.get_probe()

            payload = Namespace(features=batch, **outputs)
            EventManager.dispatch_event("after_predict", payload=payload)

            return payload

    def run_all(self, data_loader: LoadedContent) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ground_truth = []
        logits = []

        for batch in data_loader.dataset:
            ground_truth.append(batch.outputs)
            logits.append(self.run(batch).logits)

        return ground_truth, logits


class Evaluator(AbstractExecutor):
    def __init__(self, config: Config):
        super().__init__(config)

        self._predictor = Predictor(config=self.config)
        self._processor = PredictionProcessor(metrics=self.config.test_metrics, threshold=self.config.threshold)

    def run(self, data_loader: LoadedContent) -> Namespace:
        ground_truth, logits = self._predictor.run_all(data_loader=data_loader)
        return self._processor.compute_metrics(ground_truth=ground_truth, logits=logits)


class Pipeliner(AbstractExecutor):
    def __init__(self, config: Config):
        self.config = config

        self._trainer = Trainer(self.config)
        self._processor = PredictionProcessor(metrics=self.config.test_metrics, threshold=self.config.threshold)
        self._predictor = None

    def initialize_predictor(self) -> "Pipeliner":
        self._predictor = Predictor(config=self.config)
        return self

    def train(self, data_loader: LoadedContent, val_loader: Optional[LoadedContent] = None) -> None:
        self._trainer.run(data_loader=data_loader, val_loader=val_loader)

    def evaluate(self, data_loader: LoadedContent) -> Namespace:
        ground_truth, logits = self.predict(data_loader=data_loader)
        return self._processor.compute_metrics(ground_truth=ground_truth, logits=logits)

    def predict(self, data_loader: LoadedContent) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._predictor.run_all(data_loader=data_loader)

    def evaluate_all(self, data_loader: LoadedContent) -> List[Namespace]:
        results = []

        for checkpoint_path in self.find_all_checkpoints():
            config = copy(self.config)
            config.checkpoint_path = checkpoint_path

            evaluator = Evaluator(config=config)
            results.append(evaluator.run(data_loader=data_loader))

        return results

    def find_all_checkpoints(self) -> List[str]:
        checkpoint_paths = Path(self.config.output_path).rglob("*.pt")
        checkpoint_paths = sorted([str(f) for f in checkpoint_paths], key=len)
        return checkpoint_paths

    def find_best_checkpoint(self, data_loader: LoadedContent) -> "Pipeliner":
        results = self.evaluate_all(data_loader=data_loader)

        per_target_best = Namespace.reduce(results, partial(np.argmax, axis=0))
        per_target_best = getattr(per_target_best, self.config.target_metric)
        all_checkpoints = self.find_all_checkpoints()

        self.config.checkpoint_path = all_checkpoints[np.argmax(np.bincount(per_target_best))]
        self.initialize_predictor()
        return self

    def get_network(self) -> AbstractNetwork:  # throws CheckpointNotFound Exception
        super().__init__(self.config)

        self._load_checkpoint()
        return self.network


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

        gamma = max(training_examples // self.config.batch_size, 1)
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
                    self._to_device(data)
                    self.optimizer.zero_grad()
                    outputs = self.network(data.inputs)

                    payload = Namespace(features=data, logits=outputs, extras=[])
                    EventManager.dispatch_event(event_name="before_criterion", payload=payload)

                    loss = self.criterion(payload.logits, payload.features.outputs, *payload.extras)
                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()
                    self._loss_tracker.update(loss.item())

                    learning_rate_records.append(self._get_learning_rate())
                    loss_records.append(self._loss_tracker.get())
                    if iteration % 20 == 0:
                        progress_bar.set_description(f"Loss : {loss_records[-1]:.5f}")
                    progress_bar.update(1)
        except (KeyboardInterrupt, RuntimeError):
            pass

        self._plot(learning_rate_records, loss_records)

    def _get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _plot(self, learning_rate_records: List[float], loss_records: List[float]) -> None:
        import matplotlib.pyplot as plt

        plt.plot(learning_rate_records, loss_records)
        plt.xscale("log")

        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")

        plt.savefig(Path(self.config.output_path) / "lr_finder_results.png")
