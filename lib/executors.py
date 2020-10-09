import logging
from abc import ABCMeta
from typing import Tuple, NamedTuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch_geometric.data import Batch

from lib.config import Config
from lib.data_loaders import AbstractLoader
from lib.helpers import Timer
from lib.modules import WeightedBinaryCrossEntropyLoss


class AbstractExecutor(metaclass=ABCMeta):

    def __init__(self, config: Config):
        self._config = config

        self._timer = Timer()
        self._start_epoch = 0

    def load_network(self, in_features: int, out_features: int) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = self._config.get_model()
        network = model(
            in_features=in_features, hidden_features=self._config.hidden_layer_size,
            out_features=out_features, dropout=self._config.dropout
        )

        devices = self._config.enabled_gpus if self._config.use_cuda else []
        network = torch.nn.DataParallel(network, device_ids=devices)

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
            optimizer.load_state_dict(info["optimizer"])
            self._start_epoch = info["epoch"]

        return network, optimizer


class Trainer(AbstractExecutor):

    def run(self, data_loader: AbstractLoader):

        criterion = WeightedBinaryCrossEntropyLoss().to(self._config.get_device())
        network, optimizer = self.load_network(
            in_features=data_loader.get_feature_count(),
            out_features=data_loader.get_class_count()
        )

        network = network.train()
        logging.debug(network)

        dataset_size = data_loader.get_size()
        for epoch in range(self._start_epoch + 1, self._config.epochs + 1):

            accumulated_loss = 0.0
            for iteration, data in enumerate(iter(data_loader), start=1):
                optimizer.zero_grad()
                outputs = network(data)

                is_non_empty = data.y == data.y
                weights = is_non_empty.float()
                labels = data.y
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

    def __init__(self, config: Config, in_features: int, out_features: int):
        super().__init__(config)
        self._loaded_model = self.load(in_features, out_features)

    def load(self, in_features: int, out_features: int):
        if self._config.checkpoint_path is None:
            raise FileNotFoundError("No 'checkpoint_path' specified.")

        model, _ = self.load_network(in_features, out_features)
        model = model.eval()

        return model

    @torch.no_grad()
    def run(self, batch: Batch) -> torch.Tensor:

        outputs = self._loaded_model(batch)
        return outputs


class Evaluator(AbstractExecutor):

    class Results(NamedTuple):
        accuracy: float
        roc_auc_score: float
        average_precision: float

    def run(self, data_stream: AbstractLoader) -> Results:

        predictor = Predictor(
            config=self._config,
            in_features=data_stream.get_feature_count(),
            out_features=data_stream.get_class_count()
        )

        ground_truths = []
        logits = []

        for batch in iter(data_stream):
            ground_truths.extend(batch.y.cpu().detach().tolist())

            results = predictor.run(batch)
            results = results.cpu().detach().tolist()
            logits.extend(results)

        ground_truths = np.array(ground_truths).transpose()
        logits = np.array(logits).transpose()

        accuracies = []
        roc_auc_scores = []
        average_precisions = []
        for target in range(ground_truths.shape[0]):
            mask = ground_truths[target] != ground_truths[target]  # Missing Values (NaN)

            labels = np.delete(ground_truths[target], mask)
            scores = np.delete(logits[target], mask)
            predictions = np.where(scores < self._config.threshold, 0, 1)

            accuracies.append(accuracy_score(labels, predictions))
            roc_auc_scores.append(roc_auc_score(labels, scores))
            average_precisions.append(average_precision_score(labels, scores))

        return Evaluator.Results(
            accuracy=np.mean(accuracies),
            roc_auc_score=np.mean(roc_auc_scores),
            average_precision=np.mean(average_precisions)
        )
