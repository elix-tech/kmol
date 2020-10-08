import logging
from argparse import ArgumentParser
from glob import glob

from lib.config import Config
from lib.executors import Trainer, Evaluator, Predictor
from lib.data_loaders import MoleculeNetLoader


class Executor:

    def __init__(self, config: Config):
        self.__config = config

    def run(self, job: str):
        if not hasattr(self, job):
            raise ValueError("Unknown job requested: {}".format(job))

        getattr(self, job)()

    def train(self):
        data_loader = MoleculeNetLoader(config=self.__config)

        trainer = Trainer(self.__config)
        trainer.run(data_loader)

    def eval(self):
        data_loader = MoleculeNetLoader(config=self.__config)

        evaluator = Evaluator(self.__config)
        results = evaluator.run(data_loader)

        logging.info(results)

    def analyze(self):
        checkpoints = glob(self.__config.output_path + "*")
        checkpoints = sorted(checkpoints)
        checkpoints = sorted(checkpoints, key=len)

        for checkpoint in checkpoints:
            self.__config = self.__config._replace(checkpoint_path=checkpoint)
            self.eval()

    def predict(self):
        data_loader = MoleculeNetLoader(config=self.__config)
        predictor = Predictor(
            config=self.__config,
            in_features=data_loader.get_feature_count(),
            out_features=data_loader.get_class_count()
        )

        for batch in data_loader:
            predictions = predictor.run(batch)
            predictions = predictions.cpu().detach().numpy()

            print(predictions)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    executor = Executor(Config.load(args.config))
    executor.run(args.job)
