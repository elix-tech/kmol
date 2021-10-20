import json
import logging
import math
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Callable, Optional, NamedTuple, Tuple, Iterable, Any, Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, jaccard_score, roc_curve
)

from ..core.helpers import Namespace


class MetricType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class MetricConfiguration(NamedTuple):
    type: MetricType
    calculator: Callable
    uses_threshold: bool = False
    maximize: bool = True


class CustomMetrics:

    @staticmethod
    def __get_ranks(array: List[float]) -> np.ndarray:
        _, ranks, counts = np.unique(array, return_inverse=True, return_counts=True)

        clone = ranks.copy()
        cumulative_sum = -1

        for index, count in enumerate(counts):
            cumulative_sum += count
            ranks[np.where(clone == index)[0]] = cumulative_sum

        return ranks

    @staticmethod
    def pearson_correlation_coefficient(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.pearsonr(ground_truth, predictions)[0]

    @staticmethod
    def spearman_correlation_coefficient(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.spearmanr(ground_truth, predictions).correlation

    @staticmethod
    def kullback_leibler_divergence(ground_truth: List[float], predictions: List[float]) -> float:
        return stats.entropy(ground_truth, predictions)

    @staticmethod
    def jensen_shannon_divergence(ground_truth: List[float], predictions: List[float]) -> float:
        return math.exp(distance.jensenshannon(ground_truth, predictions))

    @staticmethod
    def rank_quality(ground_truth: List[float], predictions: List[float]) -> float:
        """
        For this metric, values don't matter, only the order.
        This helps if you plan to use a regression model for ranking, the exact values are not important.
        The best values is 1, the worst is 0.
        """
        if len(ground_truth) == 1:
            return 1.

        ground_truth = CustomMetrics.__get_ranks(ground_truth)
        predictions = CustomMetrics.__get_ranks(predictions)

        if len(ground_truth) == 2:
            return float(ground_truth[0] == predictions[0])

        samples_count = ground_truth.shape[0]
        worst_possible_outcome = np.arange(int(samples_count % 2 == 0), samples_count, step=2).sum() * 2

        return 1 - np.sum(np.abs(ground_truth - predictions)) / worst_possible_outcome


class AvailableMetrics:
    MAE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_absolute_error, maximize=False)
    MSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=mean_squared_error, maximize=False)
    RMSE = MetricConfiguration(type=MetricType.REGRESSION, calculator=partial(mean_squared_error, squared=False), maximize=False)
    R2 = MetricConfiguration(type=MetricType.REGRESSION, calculator=r2_score)
    PEARSON = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.pearson_correlation_coefficient)
    SPEARMAN = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.spearman_correlation_coefficient)
    KL_DIV = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.kullback_leibler_divergence, maximize=False)
    JS_DIV = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.jensen_shannon_divergence, maximize=False)
    CHEBYSHEV = MetricConfiguration(type=MetricType.REGRESSION, calculator=distance.chebyshev, maximize=False)
    MANHATTAN = MetricConfiguration(type=MetricType.REGRESSION, calculator=distance.cityblock, maximize=False)
    RANK_QUALITY = MetricConfiguration(type=MetricType.REGRESSION, calculator=CustomMetrics.rank_quality)

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
        if threshold is None:
            return np.array(cls.detach(logits))

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
                    if not metric_settings.maximize:
                        computed_value *= -1
                except ValueError:
                    computed_value = self._error_value

                metrics[metric_name].append(computed_value)

        return Namespace(**metrics)

    def find_best_threshold(self, ground_truth: List[torch.Tensor], logits: List[torch.Tensor]) -> List[float]:
        logits = [torch.sigmoid(tensor) for tensor in logits]
        ground_truth, logits, _ = self._prepare(ground_truth=ground_truth, logits=logits)

        best = []
        for i in range(len(ground_truth)):
            false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth[i], logits[i])
            best.append(thresholds[np.argmax(true_positive_rate - false_positive_rate)])

        return best


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
