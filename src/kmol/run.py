import json
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from rich import progress as pb

from mila.factories import AbstractExecutor
from .core.config import Config
from .core.helpers import Namespace, ConfidenceInterval, SuperFactory
from .core.logger import LOGGER as logging
from .core.tuning import OptunaTemplateParser
from .data.resources import DataPoint
from .data.loaders import AbstractLoader
from .data.streamers import GeneralStreamer, SubsetStreamer, CrossValidationStreamer
from .model.executors import Predictor, ThresholdFinder, LearningRareFinder, Pipeliner
from .model.metrics import PredictionProcessor, CsvLogger


class Executor(AbstractExecutor):
    def __init__(self, config: Config, config_path: Optional[str] = ""):
        super().__init__(config)
        self._config_path = config_path

    def __log_results(
        self,
        results: Namespace,
        labels: List[str],
        statistics: Tuple[Callable, ...] = (np.min, np.max, np.mean, np.median, np.std),
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

    def __revert_transformers(self, predictions: np.ndarray, streamer: GeneralStreamer):
        data = DataPoint(outputs=predictions.tolist())
        streamer.reverse_transformers(data)

        return data.outputs

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
            return 0.0

    def train(self):
        if self._config.subset:
            streamer = SubsetStreamer(config=self._config)
            Pipeliner(config=self._config).train(
                data_loader=streamer.get(
                    split_name=self._config.train_split,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                    subset_id=self._config.subset["id"],
                    subset_distributions=self._config.subset["distribution"],
                    mode=GeneralStreamer.Mode.TRAIN,
                )
            )
        else:
            streamer = GeneralStreamer(config=self._config)
            train_loader = streamer.get(
                split_name=self._config.train_split,
                batch_size=self._config.batch_size,
                shuffle=True,
                mode=GeneralStreamer.Mode.TRAIN,
            )
            val_loader = None
            if self._config.validation_split in streamer.splits:
                val_loader = streamer.get(
                    split_name=self._config.validation_split,
                    batch_size=self._config.batch_size,
                    shuffle=False,
                    mode=GeneralStreamer.Mode.TEST,
                )
            Pipeliner(config=self._config).train(data_loader=train_loader, val_loader=val_loader)

    def eval(self):
        streamer = GeneralStreamer(config=self._config)
        results = (
            Pipeliner(config=self._config)
            .initialize_predictor()
            .evaluate(
                data_loader=streamer.get(
                    split_name=self._config.test_split,
                    batch_size=self._config.batch_size,
                    shuffle=False,
                    mode=GeneralStreamer.Mode.TEST,
                )
            )
        )

        self.__log_results(results=results, labels=streamer.labels)
        return results

    def analyze(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split,
            batch_size=self._config.batch_size,
            shuffle=False,
            mode=GeneralStreamer.Mode.TEST,
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
            config = self._config.cloned_update(output_path=output_path)
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=GeneralStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                )
            )

            # evaluate all checkpoints for the current fold
            fold_results = pipeliner.evaluate_all(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=GeneralStreamer.Mode.TEST,
                    batch_size=self._config.batch_size,
                    shuffle=False,
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
            config = self._config.cloned_update(output_path=output_path)
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=GeneralStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                )
            )

            test_loader = streamer.get(
                split_name=streamer.get_fold_name(fold),
                mode=GeneralStreamer.Mode.TEST,
                batch_size=self._config.batch_size,
                shuffle=False,
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
        self._config.overwrite_checkpoint = False
        streamer = CrossValidationStreamer(config=self._config)
        folds = {}

        for fold in range(self._config.cross_validation_folds):
            output_path = "{}/.{}/".format(self._config.output_path, fold)
            config = self._config.cloned_update(output_path=output_path)
            pipeliner = Pipeliner(config=config)

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=GeneralStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                )
            )

            folds[pipeliner] = streamer.get(
                split_name=streamer.get_fold_name(fold),
                mode=GeneralStreamer.Mode.TEST,
                batch_size=self._config.batch_size,
                shuffle=False,
            )

        processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)
        results = []

        for checkpoint_id in range(1, self._config.epochs + 1):
            ground_truth = []
            logits = []

            for pipeliner, test_loader in folds.items():
                pipeliner.config.checkpoint_path = "{}/checkpoint.{}.pt".format(pipeliner.config.output_path, checkpoint_id)
                pipeliner.initialize_predictor()
                fold_ground_truth, fold_logits = pipeliner.predict(data_loader=test_loader)
                ground_truth.extend(fold_ground_truth)
                logits.extend(fold_logits)

            results.append(processor.compute_metrics(ground_truth=ground_truth, logits=logits))

        results = Namespace.max(results)
        self.__log_results(results=results, labels=streamer.labels)

        return results

    def standard_cv(self) -> Namespace:
        """
        for each fold:
            train the fold
            find the best checkpoint
            run inference on the test data (concatenating the output)
            compute metrics on the fold
        return statistics over folds
        """
        streamer = CrossValidationStreamer(config=self._config)
        processor = PredictionProcessor(metrics=self._config.test_metrics, threshold=self._config.threshold)
        all_results = []

        for fold in range(self._config.cross_validation_folds):
            output_path = "{}/.{}/".format(self._config.output_path, fold)
            config = self._config.cloned_update(output_path=output_path)
            pipeliner = Pipeliner(config=config)

            test_loader = streamer.get(
                split_name=streamer.get_fold_name(fold),
                mode=CrossValidationStreamer.Mode.TEST,
                batch_size=self._config.batch_size,
                shuffle=False,
            )

            pipeliner.train(
                data_loader=streamer.get(
                    split_name=streamer.get_fold_name(fold),
                    mode=CrossValidationStreamer.Mode.TRAIN,
                    batch_size=self._config.batch_size,
                    shuffle=True,
                ),
                val_loader=test_loader,
            )

            pipeliner.find_best_checkpoint(data_loader=test_loader)
            fold_ground_truth, fold_logits = pipeliner.predict(data_loader=test_loader)
            all_results.append(processor.compute_metrics(ground_truth=fold_ground_truth, logits=fold_logits))

        # reduction on all fold summaries
        results = Namespace.reduce(all_results, ConfidenceInterval.compute)
        self.__log_results(
            results=results,
            labels=streamer.labels,
            statistics=(np.min, np.max, np.mean, np.median),
        )

        return results

    def _collect_predictions(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.test_split,
            batch_size=self._config.batch_size,
            shuffle=False,
            mode=GeneralStreamer.Mode.TEST,
        )
        predictor = Predictor(config=self._config)
        transformer_reverter = partial(self.__revert_transformers, streamer=streamer)


        if len(self._config.prediction_additional_columns) > 0:
            loader = SuperFactory.create(AbstractLoader, self._config.loader)

        results = defaultdict(list)
        outputs_to_save = defaultdict(list)
        for batch in data_loader.dataset:
            outputs = predictor.run(batch)
            logits = outputs.logits
            variance = getattr(outputs, "logits_var", None)
            softmax_score = getattr(outputs, "softmax_score", None)
            belief_mass = getattr(outputs, "belief_mass", None)
            likelihood_ratio = getattr(outputs, "likelihood_ratio", None)
            protein_gradients = getattr(outputs, "protein_gd_mean", None)
            ligand_gradients = getattr(outputs, "ligand_gd_mean", None)
            hidden_layer_output = getattr(outputs, "hidden_layer", None)
            labels = batch.outputs.cpu().numpy()
            labels = np.apply_along_axis(transformer_reverter, axis=1, arr=labels)

            if hidden_layer_output is not None:
                outputs_to_save["hidden_layer"].extend(hidden_layer_output.cpu().numpy())

            predictions = PredictionProcessor.apply_threshold(logits, self._config.threshold)
            predictions = np.apply_along_axis(transformer_reverter, axis=1, arr=predictions)

            results["predictions"].extend(predictions)
            results["id"].extend(batch.ids)
            outputs_to_save["id"].extend(batch.ids)
            results["labels"].extend(labels)

            if variance is not None:
                results["variance"].extend(variance.cpu().numpy())
            if softmax_score is not None:
                results["softmax_score"].extend(softmax_score.cpu().numpy())
            if belief_mass is not None:
                results["belief_mass"].extend(belief_mass.cpu().numpy())
            if protein_gradients is not None:
                results["protein_gd"].extend(protein_gradients.cpu().numpy())
            if ligand_gradients is not None:
                results["ligand_gd"].extend(ligand_gradients.cpu().numpy())
            if likelihood_ratio is not None:
                results["likelihood_ratio"].extend(likelihood_ratio.cpu().numpy())

            if len(self._config.prediction_additional_columns) > 0:
                for col_name in self._config.prediction_additional_columns:
                    results[col_name].extend(loader._dataset.iloc[batch.ids][col_name].values)

        results["predictions"] = np.vstack(results["predictions"])
        results["labels"] = np.vstack(results["labels"])
        if "variance" in results:
            results["variance"] = np.vstack(results["variance"])
        if "softmax_score" in results:
            results["softmax_score"] = np.vstack(results["softmax_score"])
        if "belief_mass" in results:
            results["belief_mass"] = np.vstack(results["belief_mass"])
        if "protein_gd" in results:
            results["protein_gd"] = np.vstack(results["protein_gd"])
        if "ligand_gd" in results:
            results["ligand_gd"] = np.vstack(results["ligand_gd"])
        if "likelihood_ratio" in results:
            results["likelihood_ratio"] = np.vstack(results["likelihood_ratio"])
        if "hidden_layer" in outputs_to_save:
            outputs_to_save["hidden_layer"] = np.vstack(outputs_to_save["hidden_layer"]).tolist()

        return results, outputs_to_save, streamer.labels

    def predict(self) -> List[List[float]]:
        results, outputs_to_save, labels = self._collect_predictions()
        columns = ["id"]
        n_outputs = results["predictions"].shape[1]
        labels = labels if len(labels) == n_outputs else []

        for i in range(n_outputs):
            label = labels[i] if len(labels) else i
            results[label] = results["predictions"][:, i]
            columns.append(label)
            if len(labels):
                results[f"{label}_ground_truth"] = results["labels"][:, i]
                columns.append(f"{label}_ground_truth")
            if "variance" in results:
                results[f"{label}_logits_var"] = results["variance"][:, i]
                columns.append(f"{label}_logits_var")
            if "softmax_score" in results:
                results[f"{label}_softmax"] = results["softmax_score"][:, i]
                columns.append(f"{label}_softmax")
            if "belief_mass" in results:
                results[f"{label}_belief_mass"] = results["belief_mass"][:, i]
                columns.append(f"{label}_belief_mass")
            if "protein_gd" in results:
                results[f"{label}_protein_gd"] = results["protein_gd"][:, i]
                columns.append(f"{label}_protein_gd")
            if "ligand_gd" in results:
                results[f"{label}_ligand_gd"] = results["ligand_gd"][:, i]
                columns.append(f"{label}_ligand_gd")
            # TODO: check if this should be outside the for loop
            columns += self._config.prediction_additional_columns
        
        if "likelihood_ratio" in results:
            results["likelihood_ratio"] = results["likelihood_ratio"].flatten()
            columns.append("likelihood_ratio")
        
        results = pd.DataFrame.from_dict({c: results[c] for c in columns})

        predictions_dir = Path(self._config.output_path)
        output_file = predictions_dir / "predictions.csv"
        results.to_csv(output_file, index=False)

        logging.info(f"Predictions saved to {str(output_file)}")

        if len(outputs_to_save) > 1:
            output_file = predictions_dir / "saved_outputs.pkl"
            with output_file.open("wb") as f:
                pickle.dump(outputs_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Additional outputs saved to {str(output_file)}")

        return results

    def optimize(self) -> optuna.Study:
        if not self._config_path:
            raise AttributeError("Cannot optimize. No configuration path specified.")

        def log_summary(study):
            logging.info("---------------------------- [BEST VALUE] ----------------------------")
            logging.info(study.best_value)
            logging.info("---------------------------- [BEST TRIAL] ---------------------------- ")
            logging.info(study.best_trial)
            logging.info("---------------------------- [BEST PARAMS] ----------------------------")
            logging.info(study.best_params)

        with OptunaTemplateParser(
            config=self._config,
            evaluator=self.__run_trial,
            delete_checkpoints=True,
            log_path=str(Path(self._config.output_path) / "summary.csv"),
        ) as template_parser:

            study = optuna.create_study(direction="maximize")
            if self._config.optuna_init:
                study.enqueue_trial(self._config.optuna_init)
            try:
                study.optimize(template_parser.objective, n_trials=self._config.optuna_trials)
            except KeyboardInterrupt:
                log_summary(study)
                exit(0)

            log_summary(study)
        return study

    def find_best_checkpoint(self) -> str:
        streamer = GeneralStreamer(config=self._config)
        Pipeliner(self._config).find_best_checkpoint(
            data_loader=streamer.get(
                split_name=self._config.test_split,
                batch_size=self._config.batch_size,
                shuffle=False,
                mode=GeneralStreamer.Mode.TEST,
            )
        )

        logging.info("-----------------------------------------------------------------------")
        logging.info("Best checkpoint: {}".format(self._config.checkpoint_path))
        self.eval()

        return self._config.checkpoint_path

    def find_threshold(self) -> List[float]:
        if not self._config.checkpoint_path:
            self.find_best_checkpoint()

        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.train_split,
            batch_size=self._config.batch_size,
            shuffle=False,
            mode=GeneralStreamer.Mode.TEST,
        )

        evaluator = ThresholdFinder(self._config)
        threshold = evaluator.run(data_loader)

        logging.info("Best Thresholds: {}".format(threshold))
        logging.info("Average: {}".format(np.mean(threshold)))

        return threshold

    def find_learning_rate(self):
        streamer = GeneralStreamer(config=self._config)
        data_loader = streamer.get(
            split_name=self._config.train_split,
            batch_size=self._config.batch_size,
            shuffle=False,
            mode=GeneralStreamer.Mode.TEST,
        )

        trainer = LearningRareFinder(self._config)
        trainer.run(data_loader=data_loader)

    def visualize(self):
        from .visualization.models import IntegratedGradientsExplainer
        from .visualization.umap import UMAPVisualizer

        visualizer_params = self._config.visualizer
        visualizer_type = visualizer_params.pop("type", "umap")

        if visualizer_type not in ["umap", "iig"]:
            raise ValueError(f"Visualizer type should be one of 'umap', 'iig', received: {visualizer_type}")

        if visualizer_type == "iig":

            streamer = GeneralStreamer(config=self._config)
            data_loader = streamer.get(
                split_name=self._config.test_split,
                batch_size=1,
                shuffle=False,
                mode=GeneralStreamer.Mode.TEST,
            )
            pipeliner = Pipeliner(config=self._config)
            network = pipeliner.get_network()

            task_type = visualizer_params["is_binary_classification"]
            is_multitask = visualizer_params["is_multitask"]

            with IntegratedGradientsExplainer(
                network, self._config, is_binary_classification=task_type, is_multitask=is_multitask
            ) as visualizer:
                with pb.Progress(
                    "[progress.description]{task.description}",
                    pb.BarColumn(),
                    pb.MofNCompleteColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    pb.TimeRemainingColumn(),
                    pb.TimeElapsedColumn(),
                ) as progress:
                    for sample_id, batch in enumerate(progress.track(data_loader.dataset)):
                        pipeliner._to_device(batch)
                        for target_id in self._config.visualizer["targets"]:
                            save_path = "sample_{}_target_{}.png".format(sample_id, target_id)
                            visualizer.visualize(batch, target_id, save_path)

        elif visualizer_type == "umap":

            self._config.probe_layer = "last_hidden"
            results, outputs_to_save, labels_names = self._collect_predictions()
            hidden_features = np.array(outputs_to_save["hidden_layer"])

            label_index = visualizer_params.pop("label_index", 0)
            label_prefix = visualizer_params.pop("labels", "predictions")
            labels = results[label_prefix]
            labels = labels[:, label_index]
            label_name = labels_names[label_index] if label_prefix == "labels" else label_prefix

            visualizer = UMAPVisualizer(self._config.output_path, **visualizer_params)
            visualizer.visualize(hidden_features, labels=labels, label_name=label_name)

    def preload(self) -> None:
        GeneralStreamer(config=self._config)

    def splits(self) -> Dict[str, List[Union[int, str]]]:
        streamer = GeneralStreamer(config=self._config)

        for split_name, split_values in streamer.splits.items():
            logging.info(split_name)
            logging.info("-------------------------------------")
            logging.info(split_values)
            logging.info("")

        return streamer.splits

    def print_cfg(self) -> None:
        logging.info(json.dumps(self._config.__dict__, indent=2))


def main():
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    Executor(config=Config.from_file(args.config, args.job), config_path=args.config).run(args.job)


if __name__ == "__main__":
    main()
