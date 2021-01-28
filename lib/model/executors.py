import logging
from abc import ABCMeta
from typing import Tuple, NamedTuple, List, Callable, Union, Dict, Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.nn.modules.loss import _Loss as AbstractCriterion
from torch.optim import Optimizer as AbstractOptimizer
from torch.optim.lr_scheduler import _LRScheduler as AbstractLearningRateScheduler
from torch.utils.data import DataLoader as TorchDataLoader

from lib.core.config import Config
from lib.core.exceptions import CheckpointNotFound
from lib.core.helpers import Timer, SuperFactory
from lib.data.resources import Batch
from lib.model.architectures import AbstractNetwork


class AbstractExecutor(metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config
        self._timer = Timer()
        self._start_epoch = 0

        self._network = SuperFactory.create(AbstractNetwork, self._config.model)

        self._optimizer = None
        self._criterion = None
        self._scheduler = None

    def _load_checkpoint(self, info: Dict[str, Any]) -> None:
        self._network.load_state_dict(info["model"])

        if self._optimizer and "optimizer" in info:
            self._optimizer.load_state_dict(info["optimizer"])

        if self._scheduler and "scheduler" in info:
            self._scheduler.load_state_dict(info["scheduler"])

        if "epoch" in info:
            self._start_epoch = info["epoch"]

    def _load_network(self) -> None:
        if self._config.should_parallelize():
            self._network = torch.nn.DataParallel(self._network, device_ids=self._config.enabled_gpus)

        self._network.to(self._config.get_device())

        if self._config.checkpoint_path is not None:
            logging.info("Restoring from Checkpoint: {}".format(self._config.checkpoint_path))

            info = torch.load(self._config.checkpoint_path, map_location=self._config.get_device())
            self._load_checkpoint(info)


class Trainer(AbstractExecutor):

    def run(self, data_loader: TorchDataLoader):

        self._load_network()
        self._criterion = SuperFactory.create(AbstractCriterion, self._config.criterion).to(self._config.get_device())

        self._optimizer = SuperFactory.create(AbstractOptimizer, self._config.optimizer, {
            "params": self._network.parameters()
        })
        self._scheduler = SuperFactory.create(AbstractLearningRateScheduler, self._config.scheduler, {
            "optimizer": self._optimizer,
            "steps_per_epoch": len(data_loader.dataset) // self._config.batch_size
        })

        network = self._network.train()
        logging.debug(network)

        dataset_size = len(data_loader)
        for epoch in range(self._start_epoch + 1, self._config.epochs + 1):

            accumulated_loss = 0.0
            data: Batch
            for iteration, data in enumerate(iter(data_loader), start=1):
                self._optimizer.zero_grad()
                outputs = network(data.inputs)

                is_non_empty = data.outputs == data.outputs
                weights = is_non_empty.float()
                labels = data.outputs
                labels[~is_non_empty] = 0

                loss = self._criterion(outputs, labels, weights)
                loss.backward()

                self._optimizer.step()

                accumulated_loss += loss.item()
                if iteration % self._config.log_frequency == 0:
                    self.log(epoch, iteration, accumulated_loss, dataset_size)
                    accumulated_loss = 0.0

            self.save(epoch)

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

    def log(self, epoch: int, iteration: int, loss: float, dataset_size: int) -> None:
        processed_samples = iteration * self._config.batch_size

        logging.info(
            "epoch: {} - iteration: {} - examples: {} - loss: {} - time elapsed: {} - progress: {}".format(
                epoch,
                iteration,
                processed_samples,
                loss / self._config.log_frequency,
                str(self._timer),
                round(processed_samples / dataset_size, 4)
            )
        )


class Predictor(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

        if self._config.checkpoint_path is None:
            raise CheckpointNotFound("No 'checkpoint_path' specified.")

        self._load_network()
        self._network = self._network.eval()

    @torch.no_grad()
    def run(self, batch: Batch) -> torch.Tensor:
        return self._network(batch.inputs)


class Evaluator(AbstractExecutor):

    class Results(NamedTuple):
        accuracy: Union[List[float], np.ndarray]
        roc_auc_score: Union[List[float], np.ndarray]
        average_precision: Union[List[float], np.ndarray]

        def compute(self, callable_: Callable) -> "Evaluator.Results":
            return Evaluator.Results(
                accuracy=callable_(self.accuracy),
                roc_auc_score=callable_(self.roc_auc_score),
                average_precision=callable_(self.average_precision)
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self._predictor = Predictor(config=self._config)

    def _get_predictions(self, data_loader: TorchDataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ground_truth_cache = []
        logits_cache = []
        predictions_cache = []

        for batch in iter(data_loader):
            ground_truth_cache.extend(batch.outputs.cpu().detach().tolist())

            logits = self._predictor.run(batch)
            predictions = torch.sigmoid(logits)

            logits = logits.cpu().detach().tolist()
            logits_cache.extend(logits)

            predictions = predictions.cpu().detach().tolist()
            predictions_cache.extend(predictions)

        return (
            np.array(ground_truth_cache).transpose(),
            np.array(logits_cache).transpose(),
            np.array(predictions_cache).transpose()
        )

    def _get_metrics(
            self, ground_truth_cache: np.ndarray, logits_cache: np.ndarray, predictions_cache: np.ndarray
    ) -> Tuple[List[float], List[float], List[float]]:

        accuracies = []
        roc_auc_scores = []
        average_precisions = []
        for target in range(ground_truth_cache.shape[0]):
            mask = ground_truth_cache[target] != ground_truth_cache[target]  # Missing Values (NaN)

            ground_truth = np.delete(ground_truth_cache[target], mask)
            logits = np.delete(logits_cache[target], mask)

            predictions = np.delete(predictions_cache[target], mask)
            predictions = np.where(predictions < self._config.threshold, 0, 1)

            accuracies.append(accuracy_score(ground_truth, predictions))
            roc_auc_scores.append(roc_auc_score(ground_truth, logits))
            average_precisions.append(average_precision_score(ground_truth, logits))

        return accuracies, roc_auc_scores, average_precisions

    def run(self, data_loader: TorchDataLoader) -> Results:

        ground_truth, logits, predictions = self._get_predictions(data_loader)
        accuracies, roc_auc_scores, average_precisions = self._get_metrics(ground_truth, logits, predictions)

        return Evaluator.Results(
            accuracy=accuracies,
            roc_auc_score=roc_auc_scores,
            average_precision=average_precisions
        )
