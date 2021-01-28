from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch

from lib.core.config import Config
from lib.data.streamers import GeneralStreamer
from lib.model.executors import Trainer, Evaluator, Predictor
from mila.factories import AbstractExecutor


class Executor(AbstractExecutor):

    def __init__(self, config: Config):
        super().__init__(config)

    def train(self):
        streamer = GeneralStreamer(config=self._config, split_name=self._config.train_split)
        data_loader = streamer.get(batch_size=self._config.batch_size, shuffle=True)

        trainer = Trainer(self._config)
        trainer.run(data_loader=data_loader)

    def eval(self) -> Evaluator.Results:
        streamer = GeneralStreamer(config=self._config, split_name=self._config.test_split)
        data_loader = streamer.get(batch_size=self._config.batch_size, shuffle=False)

        evaluator = Evaluator(self._config)
        results = evaluator.run(data_loader)

        print("All: {}".format(results))
        for metric in (np.min, np.max, np.mean, np.median, np.std):
            print("{}: {}".format(metric.__name__, results.compute(metric)))

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
        for batch in data_loader:
            logits = predictor.run(batch)

            predictions = torch.sigmoid(logits)
            predictions = predictions.cpu().detach().numpy()
            predictions = np.where(predictions < self._config.threshold, 0, 1)

            predictions = predictions.astype("str")
            for prediction in predictions.tolist():
                print(",".join(prediction))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    executor = Executor(Config.from_json(args.config))
    executor.run(args.job)
