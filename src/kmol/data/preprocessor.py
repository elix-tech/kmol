import itertools
import os
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Dict, Tuple
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from torch.utils.data import Subset

from .static_augmentation import AbstractStaticAugmentation
from .featurizers import AbstractFeaturizer
from .loaders import AbstractLoader, ListLoader
from .resources import DataPoint
from .transformers import AbstractTransformer
from ..core.config import Config
from ..core.exceptions import FeaturizationError
from ..core.helpers import CacheDiskList, SuperFactory, CacheManager
from ..core.logger import LOGGER as logging
from ..core.utils import progress_bar


class AbstractPreprocessor(metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self._config = config

        self.online = self._config.online_preprocessing
        self._use_disk = self._config.preprocessing_use_disk
        self._cache_manager = CacheManager(cache_location=self._config.cache_location)

        self._featurizers = [SuperFactory.create(AbstractFeaturizer, featurizer) for featurizer in self._config.featurizers]

        self._transformers = [
            SuperFactory.create(AbstractTransformer, transformer) for transformer in self._config.transformers
        ]

        self._static_augmentations: List[AbstractStaticAugmentation] = [
            SuperFactory.create(AbstractStaticAugmentation, s_augmentation)
            for s_augmentation in self._config.static_augmentations
        ]

    def preprocess(self, sample):
        self._featurize(sample)
        self._apply_transformers(sample)
        return sample

    def _featurize(self, sample: DataPoint):
        for featurizer in self._featurizers:
            featurizer.run(sample)

    def _apply_transformers(self, sample: DataPoint) -> None:
        for transformer in self._transformers:
            transformer.apply(sample)

    def reverse_transformers(self, sample: DataPoint) -> None:
        for transformer in reversed(self._transformers):
            transformer.reverse(sample)

    def _get_chunks(self, dataset):
        n_jobs = self._config.featurization_jobs
        chunk_size = len(dataset) // n_jobs
        chunks = [list(range(i, i + chunk_size)) for i in range(0, len(dataset), chunk_size)]
        return [Subset(dataset, chunk) for chunk in chunks]

    def _run_parrallel(self, func, dataset, use_disk=False):
        chunks = self._get_chunks(dataset)
        with progress_bar() as progress:
            futures = []
            with multiprocessing.Manager() as manager:
                _progress = manager.dict()
                overall_progress_task = progress.add_task("[green]All jobs progress:")

                with ProcessPoolExecutor(max_workers=self._config.featurization_jobs) as executor:
                    for n, chunk in enumerate(chunks, 1):
                        task_id = progress.add_task(f"featurizer {n}", visible=False)
                        futures.append(executor.submit(func, _progress, task_id, chunk))

                    n_finished = 0
                    while n_finished < len(futures):
                        for task_id, update_data in _progress.items():
                            latest = update_data["progress"]
                            total = update_data["total"]
                            progress.update(task_id, completed=latest, total=total, visible=latest < total)
                        n_finished = sum([future.done() for future in futures])
                        progress.update(overall_progress_task, completed=n_finished, total=len(futures))

        if use_disk:
            logging.info("Merging cache files...")
            disk_lists = [future.result() for future in futures]
            dataset = disk_lists[0]
            with progress_bar() as progress:
                for disk_list in progress.track(disk_lists[1:]):
                    dataset.extend(disk_list)

            # clean up
            for dl in disk_lists[1:]:
                dl.clear()
        else:
            dataset = list(itertools.chain.from_iterable([future.result() for future in futures]))

        return dataset

    @abstractmethod
    def _load_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _load_augmented_data(self):
        raise NotImplementedError


class OnlinePreprocessor(AbstractPreprocessor):
    def __init__(self, config) -> None:
        super().__init__(config)

    def _load_dataset(self) -> ListLoader:
        dataset = SuperFactory.create(AbstractLoader, self._config.loader)
        self.dataset_size = len(dataset)
        self.run_preprocess(dataset)
        return dataset

    def run_preprocess(self, dataset: ListLoader):
        """
        Run preprocessor on featurizer with `_should_cache` only on unique element
        define by `compute_unique_dataset_to_cache` function in the featurizer.
        The featurizer will be cached and loaded on the next run.
        """

        def run_f(f, dataset):
            for sample in tqdm(dataset):
                f.run(sample)
            return f

        for i, f in enumerate(self._featurizers):
            if getattr(f, "_should_cache") and hasattr(f, "compute_unique_dataset_to_cache"):
                func = partial(run_f, f)
                dataset = f.compute_unique_dataset_to_cache(dataset)

                result = self._cache_manager.execute_cached_operation(
                    processor=func,
                    clear_cache=self._config.clear_cache,
                    arguments={"dataset": deepcopy(dataset)},
                    cache_key={
                        "unique_data": sorted([sample.inputs[f._inputs[0]] for sample in dataset]),
                        "featurizer": self._config.featurizers[i],
                    },
                )
                self._featurizers[i] = result

    def _load_augmented_data(self):
        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        for i, a in enumerate(self._static_augmentations):
            logging.info(f"Starting {type(a)} augmentation...")
            self._static_augmentations[i] = self._cache_manager.execute_cached_operation(
                processor=partial(a.generate_augmented_data, loader),
                clear_cache=self._config.clear_cache,
                arguments={},
                cache_key={
                    "loader": self._config.loader,
                    "static_augmentation": self._config.static_augmentations[i],
                    "last_modified": os.path.getmtime(self._config.loader["input_path"]),
                    "online": True,
                },
            )

        # Add id to augmented data.
        i = 0
        for dataset in [a.aug_dataset for a in self._static_augmentations]:
            if type(dataset) == list:
                continue
            i += len(dataset)
        for dataset in [a.aug_dataset for a in self._static_augmentations]:
            if type(dataset) == list:
                for data in dataset:
                    data.id_ = self.dataset_size + i
                    i += 1

    def _add_static_aug_dataset(self, dataset):
        # Need to transform dataset in datapoint to be able to merge the dataset
        # dataset = [data_point for data_point in dataset]
        for a in self._static_augmentations:
            if type(a.aug_dataset) == list:
                continue
            dataset = dataset + a.aug_dataset

        if any([type(a.aug_dataset) == list for a in self._static_augmentations]):
            dataset = [data_point for data_point in dataset]
        for dataset in [a.aug_dataset for a in self._static_augmentations]:
            if type(dataset) == list:
                dataset = dataset + a.aug_dataset
        return dataset


class CachePreprocessor(AbstractPreprocessor):
    def _load_dataset(self) -> AbstractLoader:
        dataset = self._cache_manager.execute_cached_operation(
            processor=self._prepare_dataset,
            clear_cache=self._config.clear_cache,
            arguments={},
            cache_key={
                "loader": self._config.loader,
                "featurizers": self._config.featurizers,
                "transformers": self._config.transformers,
                "last_modified": os.path.getmtime(self._config.loader["input_path"]),
            },
        )
        self.dataset_size = len(dataset)
        return dataset

    def _prepare_dataset(self) -> ListLoader:

        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        logging.info("Starting featurization...")
        dataset = self._run_parrallel(self._prepare_chunk, loader, self._use_disk)

        ids = [sample.id_ for sample in dataset]
        return ListLoader(dataset, ids)

    def _prepare_chunk(self, progress, task_id, loader) -> List[DataPoint]:
        dataset = CacheDiskList(tmp_dir=self._config.preprocessing_disk_dir) if self._use_disk else []
        for n, sample in enumerate(loader):
            smiles = sample.inputs["smiles"] if "smiles" in sample.inputs else ""
            try:
                sample = self.preprocess(sample)
                dataset.append(sample)

            except FeaturizationError as e:
                logging.warning(e)
            except Exception as e:
                logging.debug(f"{sample} {smiles} - {e}")

            progress[task_id] = {"progress": n + 1, "total": len(loader)}

        self.dataset_size = len(dataset)
        return dataset

    def _load_augmented_data(self) -> Tuple[ListLoader, Dict]:
        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        for i, a in enumerate(self._static_augmentations):
            self._static_augmentations[i] = self._cache_manager.execute_cached_operation(
                processor=partial(self._apply_deterministic_augmentation, a, loader),
                clear_cache=self._config.clear_cache,
                arguments={},
                cache_key={
                    "loader": self._config.loader,
                    "static_augmentation": self._config.static_augmentations[i],
                    "last_modified": os.path.getmtime(self._config.loader["input_path"]),
                },
            )

        # Add id to augmented data.
        i = 0
        for dataset in [a.aug_dataset for a in self._static_augmentations]:
            for data in dataset:
                data.id_ = self.dataset_size + i
                i += 1

    def _apply_deterministic_augmentation(self, a: AbstractStaticAugmentation, loader: AbstractLoader) -> ListLoader:
        logging.info(f"Starting {type(a)} augmentation...")
        a.generate_augmented_data(loader)
        logging.info(f"Starting Featurization of  {type(a)} augmented data...")
        # a.aug_dataset = self._prepare_chunk(a.aug_dataset)
        logging.info("Starting featurization...")
        a.aug_dataset = self._run_parrallel(self._prepare_chunk, a.aug_dataset)
        return a

    def _add_static_aug_dataset(self, dataset: ListLoader):
        tmp_dataset = dataset._dataset
        for a in self._static_augmentations:
            tmp_dataset += a.aug_dataset
        return ListLoader(tmp_dataset, range(len(tmp_dataset)))
