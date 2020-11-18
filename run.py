from argparse import ArgumentParser
from glob import glob

import numpy as np

from lib.config import Config
from lib.executors import Trainer, Evaluator, Predictor
from mila.factories import AbstractExecutor


class Executor(AbstractExecutor):

    def train(self):
        data_loader = self._config.get_data_loader(mode="train")

        trainer = Trainer(self._config)
        trainer.run(data_loader)

    def eval(self) -> Evaluator.Results:
        data_loader = self._config.get_data_loader(mode="test")

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
        data_loader = self._config.get_data_loader(mode="test")
        predictor = Predictor(config=self._config)

        for batch in data_loader:
            predictions = predictor.run(batch)
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
