import json
import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Callable, Optional, NamedTuple, Tuple, Iterable, Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, jaccard_score
)

from lib.core.helpers import Namespace


class MetricType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class MetricConfiguration(NamedTuple):
    type: MetricType
    calculator: Callable
    uses_threshold: bool = False


class AvailableMetrics:
    MAE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_absolute_error)
    MSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_squared_error)
    RMSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=partial(mean_squared_error, squared=False))
    R2 = MetricConfiguration(type=MetricType.REGRESSION, calculator=r2_score)
    ROC_AUC = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=roc_auc_score)
    PR_AUC = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=average_precision_score)
    ACCURACY = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=accuracy_score, uses_threshold=True)
    PRECISION = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=partial(precision_score, zero_division=1), uses_threshold=True)
    RECALL = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=partial(recall_score, zero_division=1), uses_threshold=True)
    F1 = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=f1_score, uses_threshold=True)
    COHEN_KAPPA = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=cohen_kappa_score, uses_threshold=True)
    JACCARD = MetricConfiguration(type=MetricType.CLASSIFICATION, calculator=jaccard_score, uses_threshold=True)


class PredictionProcessor:

    def __init__(self, metrics: List[str], threshold: Optional[float] = None, error_value: Any = None):
        self._metrics = self._map_metrics(metrics)
        self._threshold = threshold
        self._error_value = error_value

    def _map_metrics(self, metrics: List[str]) -> Dict[str, MetricConfiguration]:
        return {metric: getattr(AvailableMetrics, metric.upper()) for metric in metrics}

    def _needs_predictions(self) -> bool:
        return any(metric.uses_threshold for metric in self._metrics.values())

    @classmethod
    def detach(cls, tensor: torch.Tensor) -> List[List[float]]:
        return tensor.cpu().detach().tolist()

    @classmethod
    def apply_threshold(cls, logits: torch.Tensor, threshold: float) -> np.ndarray:
        predictions = torch.sigmoid(logits)
        predictions = cls.detach(predictions)

        return np.where(np.less(predictions, threshold), 0, 1)

    @classmethod
    def compute_statistics(
            cls, metrics: Namespace,
            statistics: Iterable[Callable] = (np.min, np.max, np.mean, np.median, np.std)
    ) -> Namespace:

        results = defaultdict(list)
        for name, values in vars(metrics).items():
            for statistic in statistics:
                results[name].append(statistic(values))

        return Namespace(**results)

    def _prepare(
            self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]
    ) -> Tuple[List[List[float]], List[List[float]], Optional[List[List[float]]]]:

        # concatenate tensors and transpose
        ground_truth = torch.cat(ground_truth).t()
        logits = torch.cat(logits).t()

        # move values to CPU, get predictions from logits if needed
        ground_truth = self.detach(ground_truth)
        predictions = None
        if self._needs_predictions():
            predictions = self.apply_threshold(logits, self._threshold).tolist()
        logits = self.detach(logits)

        # remove missing labels
        mask = np.isnan(ground_truth)
        for i in range(len(ground_truth)):
            ground_truth[i] = np.delete(ground_truth[i], mask[i])
            logits[i] = np.delete(logits[i], mask[i])

            if predictions:
                predictions[i] = np.delete(predictions[i], mask[i])

        return ground_truth, logits, predictions

    def compute_metrics(self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]) -> Namespace:
        ground_truth, logits, predictions = self._prepare(ground_truth=ground_truth, logits=logits)
        metrics = defaultdict(list)

        for target_index in range(len(ground_truth)):
            for metric_name, metric_settings in self._metrics.items():
                labels = predictions[target_index] if metric_settings.uses_threshold else logits[target_index]

                try:
                    computed_value = metric_settings.calculator(ground_truth[target_index], labels)
                except ValueError:
                    computed_value = self._error_value

                metrics[metric_name].append(computed_value)

        return Namespace(**metrics)


class AbstractMetricLogger(metaclass=ABCMeta):

    @abstractmethod
    def log_header(self, headers: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_content(self, content: Namespace) -> None:
        raise NotImplementedError


class JsonLogger(AbstractMetricLogger):

    def log_header(self, headers: List[str]) -> None:
        logging.info("--------------------------------------------------------------")
        logging.info(headers)
        logging.info("--------------------------------------------------------------")

    def log_content(self, content: Namespace) -> None:
        print(json.dumps(vars(content)))


class CsvLogger(AbstractMetricLogger):

    def log_header(self, headers: List[str]) -> None:
        logging.info("--------------------------------------------------------------")
        print("metric,{}".format(",".join(headers)))
        logging.info("--------------------------------------------------------------")

    def log_content(self, content: Namespace) -> None:
        for name, values in vars(content).items():
            values = [str(value) for value in values]
            print("{},{}".format(name, ",".join(values)))
