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

        print(results)
        return results

    def analyze(self):
        checkpoints = glob(self._config.output_path + "*")
        checkpoints = sorted(checkpoints)
        checkpoints = sorted(checkpoints, key=len)

        best = Evaluator.Results(accuracy=0, roc_auc_score=0, average_precision=0)
        for checkpoint in checkpoints:
            self._config.checkpoint_path = checkpoint
            results = self.eval()

            best = Evaluator.Results(
                accuracy=max(best.accuracy, results.accuracy),
                roc_auc_score=max(best.roc_auc_score, results.roc_auc_score),
                average_precision=max(best.average_precision, results.average_precision)
            )

        print(best)

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
