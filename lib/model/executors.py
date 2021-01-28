import logging
from abc import ABCMeta
from typing import Tuple, NamedTuple, List, Callable, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader as TorchDataLoader

from lib.core.config import Config
from lib.core.exceptions import CheckpointNotFound
from lib.core.helpers import Timer, SuperFactory
from lib.data.resources import Batch
from lib.model.architectures import AbstractNetwork
from lib.model.modules import WeightedBinaryCrossEntropyLoss


class AbstractExecutor(metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config

        self._timer = Timer()
        self._start_epoch = 0

    def load_network(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        network = SuperFactory.create(AbstractNetwork, self._config.model)
        if self._config.should_parallelize():
            network = torch.nn.DataParallel(network, device_ids=self._config.enabled_gpus)

        network.to(self._config.get_device())

        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay
        )

        if self._config.checkpoint_path is not None:
            logging.info("Restoring from Checkpoint: {}".format(self._config.checkpoint_path))
            info = torch.load(self._config.checkpoint_path, map_location=self._config.get_device())

            network.load_state_dict(info["model"])

            if "optimizer" in info:
                optimizer.load_state_dict(info["optimizer"])

            if "epoch" in info:
                self._start_epoch = info["epoch"]

        return network, optimizer


class Trainer(AbstractExecutor):

    def run(self, data_loader: TorchDataLoader):

        criterion = WeightedBinaryCrossEntropyLoss().to(self._config.get_device())
        network, optimizer = self.load_network()

        network = network.train()
        logging.debug(network)

        dataset_size = len(data_loader)
        for epoch in range(self._start_epoch + 1, self._config.epochs + 1):

            accumulated_loss = 0.0
            data: Batch
            for iteration, data in enumerate(iter(data_loader), start=1):
                optimizer.zero_grad()
                outputs = network(data.inputs)

                is_non_empty = data.outputs == data.outputs
                weights = is_non_empty.float()
                labels = data.outputs
                labels[~is_non_empty] = 0

                loss = criterion(outputs, labels, weights)
                loss.backward()

                optimizer.step()

                accumulated_loss += loss.item()
                if iteration % self._config.log_frequency == 0:
                    self.log(epoch, iteration, accumulated_loss, dataset_size)
                    accumulated_loss = 0.0

            self.save(epoch, network, optimizer)

    def save(self, epoch: int, network: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        info = {"epoch": epoch, "optimizer": optimizer.state_dict(), "model": network.state_dict()}
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
        self._loaded_model = self.load()

    def load(self):
        if self._config.checkpoint_path is None:
            raise CheckpointNotFound("No 'checkpoint_path' specified.")

        model, _ = self.load_network()
        model = model.eval()

        return model

    @torch.no_grad()
    def run(self, batch: Batch) -> torch.Tensor:
        outputs = self._loaded_model(batch.inputs)
        return outputs


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