from argparse import ArgumentParser
from glob import glob
from typing import Dict, List

import numpy as np

from lib.core.config import Config
from lib.data.streamers import GeneralStreamer
from lib.model.executors import Trainer, Evaluator, Predictor
from lib.model.metrics import PredictionProcessor, CsvLogger
from mila.factories import AbstractExecutor


class Executor(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

    def __log_validation_results(self, results: Dict[str, List[float]], labels: List[str]):
        logger = CsvLogger()

        logger.log_header(labels)
        logger.log_content(results)

        statistics = [np.min, np.max, np.mean, np.median, np.std]
        values = PredictionProcessor.compute_statistics(results)
        statistics = [statistic.__name__ for statistic in statistics]

        logger.log_header(statistics)
        logger.log_content(values)

    def train(self):
        streamer = GeneralStreamer(config=self._config, split_name=self._config.train_split)
        data_loader = streamer.get(batch_size=self._config.batch_size, shuffle=True)

        trainer = Trainer(self._config)
        trainer.run(data_loader=data_loader)

    def eval(self) -> Dict[str, List[float]]:
        streamer = GeneralStreamer(config=self._config, split_name=self._config.test_split)
        data_loader = streamer.get(batch_size=self._config.batch_size, shuffle=False)

        evaluator = Evaluator(self._config)
        results = evaluator.run(data_loader)

        self.__log_validation_results(results=results, labels=streamer.labels)
        return results

    def analyze(self):
        checkpoints = glob(self._config.output_path + "*")
        checkpoints = sorted(checkpoints)
        checkpoints = sorted(checkpoints, key=len)

        for checkpoint in checkpoints:
            self._config.checkpoint_path = checkpoint
            self.eval()

    def predict(self):
        streamer = GeneralStreamer(config=self._config, split_name=self._config.test_split)
        data_loader = streamer.get(batch_size=self._config.batch_size, shuffle=False)

        predictor = Predictor(config=self._config)
        print(",".join(streamer.labels))

        for batch in data_loader:
            logits = predictor.run(batch)

            predictions = PredictionProcessor.apply_threshold(logits, self._config.threshold)
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
