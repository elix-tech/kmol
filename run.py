import logging
import os
from argparse import ArgumentParser
from glob import glob
from typing import List

import numpy as np
from torch.utils.data import DataLoader

from lib.core.config import Config
from lib.core.helpers import Namespace
from lib.data.resources import Data
from lib.data.streamers import GeneralStreamer, CrossValidationStreamer
from lib.model.executors import Trainer, Evaluator, Predictor
from lib.model.metrics import PredictionProcessor, CsvLogger
from mila.factories import AbstractExecutor


class Executor(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

    def __log_results(self, results: Namespace, labels: List[str]):
        logger = CsvLogger()

        logger.log_header(labels)
        logger.log_content(results)

        statistics = [np.min, np.max, np.mean, np.median, np.std]
        values = PredictionProcessor.compute_statistics(results)
        statistics = [statistic.__name__ for statistic in statistics]

        logger.log_header(statistics)
        logger.log_content(values)

    def __revert_transformations(self, predictions: np.ndarray, streamer: GeneralStreamer):
        data = Data(outputs=predictions.transpose())
        streamer.reverse_transformers(data)

        return data.outputs.transpose()

    def __train(self, data_loader: DataLoader) -> None:
        trainer = Trainer(self._config)
        trainer.run(data_loader=data_loader)

    def __eval(self, data_loader: DataLoader) -> Namespace:
        evaluator = Evaluator(self._config)
        results = evaluator.run(data_loader)

        return results

    def __analyze(self, data_loader: DataLoader) -> List[Namespace]:
        checkpoints = glob(self._config.output_path + "*")
        checkpoints = sorted(checkpoints)
        checkpoints = sorted(checkpoints, key=len)

        results = []
        for checkpoint in checkpoints:
            self._config.checkpoint_path = checkpoint

            result = self.__eval(data_loader=data_loader)
            results.append(result)

        return results

    def train(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.train_split, batch_size=self._config.batch_size, shuffle=True
        )

        self.__train(data_loader=data_loader)

    def eval(self) -> Namespace:
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        )

        results = self.__eval(data_loader=data_loader)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def analyze(self) -> Namespace:
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        )

        results = self.__analyze(data_loader)
        for checkpoint_id, result in enumerate(results):
            self.__log_results(results=result, labels=streamer.labels + ["[{}]".format(checkpoint_id)])

        logging.info("============================ Best ============================")
        results = Namespace.max(results)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def cv(self) -> Namespace:
        streamer = CrossValidationStreamer(config=self._config)
        output_path = self._config.output_path

        results = []
        for fold in range(self._config.cross_validation_folds):
            self._config.checkpoint_path = None
            self._config.output_path = "{}/.{}/".format(output_path, fold)
            if not os.path.exists(self._config.output_path):
                os.makedirs(self._config.output_path)

            self.__train(
                streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True
                )
            )

            result = self.__analyze(
                streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TEST,
                    batch_size=self._config.batch_size,
                    shuffle=False
                )
            )

            results.append(Namespace.max(result))

        results = Namespace.mean(results)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def predict(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        )

        predictor = Predictor(config=self._config)
        print(",".join(streamer.labels))

        for batch in data_loader:
            logits = predictor.run(batch)

            predictions = PredictionProcessor.apply_threshold(logits, self._config.threshold)
            predictions = self.__revert_transformations(predictions, streamer)
            predictions = predictions.astype("str").tolist()

            for prediction in predictions:
                print(",".join(prediction))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    executor = Executor(Config.from_json(args.config))
    executor.run(args.job)
