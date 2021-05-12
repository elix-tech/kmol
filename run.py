import logging
from argparse import ArgumentParser
from typing import List, Tuple, Callable, Optional, Dict, Union

import joblib
import numpy as np
import optuna
from tqdm import tqdm

from lib.core.config import Config
from lib.core.helpers import Namespace, ConfidenceInterval
from lib.core.tuning import OptunaTemplateParser
from lib.data.resources import DataPoint
from lib.data.streamers import GeneralStreamer, SubsetStreamer, CrossValidationStreamer
from lib.model.executors import Predictor, ThresholdFinder, LearningRareFinder, Pipeliner
from lib.model.metrics import PredictionProcessor, CsvLogger


class Executor:

    def __init__(self, config: Config, config_path: Optional[str] = ""):
        self._config = config
        self._config_path = config_path

    def run(self, job: str):
        if not hasattr(self, job):
            raise ValueError("Unknown job requested: {}".format(job))

        getattr(self, job)()

    def __log_results(
            self, results: Namespace, labels: List[str],
            statistics: Tuple[Callable, ...] = (np.min, np.max, np.mean, np.median, np.std)
    ):
        logger = CsvLogger()

        logger.log_header(labels)
        logger.log_content(results)

        if len(labels) > 1:
            try:
                values = PredictionProcessor.compute_statistics(results, statistics=statistics)
                statistics = [statistic.__name__ for statistic in statistics]

                logger.log_header(statistics)
                logger.log_content(values)
            except TypeError:
                logging.debug("[Notice] Cannot compute statistics. Some metrics could not be computed for all targets.")

    def __revert_transformations(self, predictions: np.ndarray, streamer: GeneralStreamer):
        data = DataPoint(outputs=predictions.transpose())
        streamer.reverse_transformers(data)

        return data.outputs.transpose()

    def __run_trial(self, config: Config) -> float:
        try:
            executor = Executor(config=config)
            executor.train()

            results = executor.analyze()
            joblib.dump(results, "{}/.metrics.pkl".format(config.output_path))

            best = getattr(results, self._config.target_metric)
            return float(np.mean(best))

        except Exception as e:
            logging.error("[Trial Failed] {}".format(e))
            return 0.

    def train(self):
        if self._config.subset:
            streamer = SubsetStreamer(config=self._config)
            Pipeliner(config=self._config).train(data_loader=streamer.get(
                split_name=self._config.train_split, batch_size=self._config.batch_size, shuffle=True,
                subset_id=self._config.subset["id"], subset_distributions=self._config.subset["distribution"]
            ))
        else:
            streamer = GeneralStreamer(config=self._config)
            Pipeliner(config=self._config).train(data_loader=streamer.get(
                split_name=self._config.train_split, batch_size=self._config.batch_size, shuffle=True
            ))

    def eval(self):
        streamer = GeneralStreamer(config=self._config)
        results = Pipeliner(config=self._config).initialize_predictor().evaluate(data_loader=streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        ))

        self.__log_results(results=results, labels=streamer.labels)
        return results

    def analyze(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        )

        results = Pipeliner(config=self._config).evaluate_all(data_loader=data_loader)
        for checkpoint_id, result in enumerate(results):
            self.__log_results(results=result, labels=streamer.labels + ["[{}]".format(checkpoint_id)])

        logging.info("============================ Best ============================")
        results = Namespace.max(results)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def mean_cv(self) -> Namespace:
        """
        for each fold:
            train the fold
            evaluate the fold (keep metrics from the best checkpoint)
        aggregate results (compute metric averages and confidence interval)
        """
        streamer = CrossValidationStreamer(config=self._config)
        all_results = []

        for fold in range(self._config.cross_validation_folds):
            output_path = "{}/.{}/".format(self._config.output_path, fold)
            config = Config(**{**vars(self._config), **{"output_path": output_path}})
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True
                )
            )

            # evaluate all checkpoints for the current fold
            fold_results = pipeliner.evaluate_all(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TEST,
                    batch_size=self._config.batch_size,
                    shuffle=False
                )
            )

            # reduction on all checkpoints for a single fold
            all_results.append(Namespace.max(fold_results))

        # reduction on all fold summaries
        results = Namespace.reduce(all_results, ConfidenceInterval.compute)
        self.__log_results(results=results, labels=streamer.labels, statistics=(np.min, np.max, np.mean, np.median))

        return results

    def full_cv(self) -> Namespace:
        """
        for each fold:
            train the fold
            find the best checkpoint
            run inference on the test data (concatenating the output)
        compute metrics on the predicted values in one go
        """
        streamer = CrossValidationStreamer(config=self._config)

        ground_truth = []
        logits = []

        for fold in range(self._config.cross_validation_folds):
            output_path = "{}/.{}/".format(self._config.output_path, fold)
            config = Config(**{**vars(self._config), **{"output_path": output_path}})
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True
                )
            )

            test_loader = streamer.get(
                split_name=streamer.get_fold_name(fold),
                mode=CrossValidationStreamer.Mode.TEST,
                batch_size=self._config.batch_size,
                shuffle=False
            )

            pipeliner.find_best_checkpoint(data_loader=test_loader)
            fold_ground_truth, fold_logits = pipeliner.predict(data_loader=test_loader)

            ground_truth.extend(fold_ground_truth)
            logits.extend(fold_logits)

        processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)
        results = processor.compute_metrics(ground_truth=ground_truth, logits=logits)

        self.__log_results(results=results, labels=streamer.labels)
        return results

    def step_cv(self) -> Namespace:
        """
        for each fold:
            train the fold
        for range(checkpoint counts):
            load each fold
            run inference on the test data (concatenating the output)
            compute metrics on the predicted values in one go
        return the best checkpoint metrics
        """
        streamer = CrossValidationStreamer(config=self._config)
        folds = {}

        for fold in range(self._config.cross_validation_folds):
            output_path = "{}/.{}/".format(self._config.output_path, fold)
            config = Config(**{**vars(self._config), **{"output_path": output_path}})
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True
                )
            )

            folds[pipeliner] = streamer.get(
                split_name=streamer.get_fold_name(fold),
                mode=CrossValidationStreamer.Mode.TEST,
                batch_size=self._config.batch_size,
                shuffle=False
            )

        processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)
        results = []

        for checkpoint_id in range(1, self._config.epochs + 1):
            ground_truth = []
            logits = []

            for pipeliner, test_loader in folds.items():
                pipeliner._config.checkpoint_path = "{}/checkpoint.{}".format(
                    pipeliner._config.output_path, checkpoint_id
                )

                pipeliner.initialize_predictor()
                fold_ground_truth, fold_logits = pipeliner.predict(data_loader=test_loader)
                ground_truth.extend(fold_ground_truth)
                logits.extend(fold_logits)

            results.append(processor.compute_metrics(ground_truth=ground_truth, logits=logits))

        results = Namespace.max(results)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def predict(self) -> List[List[float]]:
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        )

        predictor = Predictor(config=self._config)
        print(",".join(streamer.labels))

        results = []
        for batch in data_loader.dataset:
            logits = predictor.run(batch)

            predictions = PredictionProcessor.apply_threshold(logits, self._config.threshold)
            predictions = self.__revert_transformations(predictions, streamer)
            results.extend(predictions)

            predictions = predictions.astype("str").tolist()
            for prediction in predictions:
                print(",".join(prediction))

        return results

    def optimize(self) -> optuna.Study:
        if not self._config_path:
            raise AttributeError("Cannot optimize. No configuration path specified.")

        template_parser = OptunaTemplateParser(
            template_path=self._config_path,
            evaluator=self.__run_trial,
            delete_checkpoints=True
        )

        study = optuna.create_study(direction='maximize')
        study.optimize(template_parser.objective, n_trials=self._config.optuna_trials)

        logging.info("---------------------------- [BEST VALUE] ----------------------------")
        logging.info(study.best_value)
        logging.info("---------------------------- [BEST TRIAL] ---------------------------- ")
        logging.info(study.best_trial)
        logging.info("---------------------------- [BEST PARAMS] ----------------------------")
        logging.info(study.best_params)

        return study

    def find_best_checkpoint(self) -> str:
        streamer = GeneralStreamer(config=self._config)
        Pipeliner(self._config).find_best_checkpoint(data_loader=streamer.get(
            split_name=self._config.test_split, batch_size=self._config.batch_size, shuffle=False
        ))

        logging.info("-----------------------------------------------------------------------")
        print("Best checkpoint: {}".format(self._config.checkpoint_path))
        self.eval()

        return self._config.checkpoint_path

    def find_threshold(self) -> List[float]:
        if not self._config.checkpoint_path:
            self.find_best_checkpoint()

        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.train_split, batch_size=self._config.batch_size, shuffle=False
        )

        evaluator = ThresholdFinder(self._config)
        threshold = evaluator.run(data_loader)

        print("Best Thresholds: {}".format(threshold))
        print("Average: {}".format(np.mean(threshold)))

        return threshold

    def find_learning_rate(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.train_split, batch_size=self._config.batch_size, shuffle=False
        )

        trainer = LearningRareFinder(self._config)
        trainer.run(data_loader=data_loader)

    def visualize(self):
        from lib.visualization.models import IntegratedGradientsExplainer

        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(split_name=self._config.test_split, batch_size=1, shuffle=False)
        network = Pipeliner(config=self._config).get_network()

        with IntegratedGradientsExplainer(network, self._config) as visualizer:
            for sample_id, batch in enumerate(tqdm(data_loader.dataset)):
                for target_id in self._config.visualizer["targets"]:
                    save_path = "sample_{}_target_{}.png".format(sample_id, target_id)
                    visualizer.visualize(batch, target_id, save_path)

    def preload(self) -> None:
        GeneralStreamer(config=self._config)

    def splits(self) -> Dict[str, List[Union[int, str]]]:
        streamer = GeneralStreamer(config=self._config)

        for split_name, split_values in streamer.splits.items():
            print(split_name)
            print("-------------------------------------")
            print(split_values)
            print("")

        return streamer.splits


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    Executor(config=Config.from_json(args.config), config_path=args.config).run(args.job)
