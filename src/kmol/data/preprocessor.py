import os
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Dict, Tuple
from copy import deepcopy
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import pickle
import logging
import traceback

import numpy as np

from kmol.data.static_augmentation import AbstractStaticAugmentation
from kmol.data.featurizers import AbstractFeaturizer
from kmol.data.loaders import AbstractLoader, ListLoader
from kmol.data.resources import DataPoint
from kmol.data.transformers import AbstractTransformer
from kmol.core.config import Config
from kmol.core.exceptions import FeaturizationError
from kmol.core.helpers import CacheDiskList, SuperFactory, CacheManager
from kmol.core.logger import LOGGER as logger
from kmol.core.utils import progress_bar


class AbstractPreprocessor(metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self._config = config

        self._cache_manager = CacheManager(cache_location=self._config.cache_location)

        self._featurizers = [SuperFactory.create(AbstractFeaturizer, featurizer) for featurizer in self._config.featurizers]

        [f.set_device(self._config.get_device()) for f in self._featurizers]

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

    def _wrapper_mp_worker(self, func, progress, sample):
        logger.stdout_handler.setLevel(self._config.log_level.upper())
        try:
            data = func(sample)
        except FeaturizationError as e:
            tb_str = traceback.format_exc()
            logger.warning(f"{e}\n{tb_str}")
            return None
        except Exception as e:
            tb_str = traceback.format_exc()
            smiles = sample.inputs["smiles"] if "smiles" in sample.inputs else ""
            logger.error(f"{sample} {smiles} - {e}\n{tb_str}")
            return None
        progress.set(progress.get() + 1)
        # We are pickling the data to avoid any mmap errors
        buffer = pickle.dumps(data, protocol=-1)
        return np.frombuffer(buffer, dtype=np.uint8)

    def _run_parrallel(self, func, loader, disk_dir=None):
        with progress_bar() as progress:
            with multiprocessing.Manager() as manager, multiprocessing.Pool(self._config.featurization_jobs) as executor:
                # Add a counter for "good" estimation of the advancement
                _progress = manager.Value("i", 0)
                func = partial(self._wrapper_mp_worker, func, _progress)
                overall_progress_task = progress.add_task("[green] Progress:", total=len(loader))

                dataset = CacheDiskList(tmp_dir=disk_dir) if disk_dir is not None else []
                for i, result in enumerate(executor.imap(func, loader, chunksize=100)):
                    progress.update(overall_progress_task, completed=max(_progress.get(), i + 1), total=len(loader))
                    if result is not None:
                        result = pickle.loads(memoryview(result))
                        dataset.append(result)

        return dataset

    @abstractmethod
    def _load_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def _load_augmented_data(self):
        raise NotImplementedError


class OnlinePreprocessor(AbstractPreprocessor):
    """
    Will run the featurization at each step of the training. No features will be
    saved. Ideal when the dataset is very large and can't be kept in memory but
    the featurization is not a bottleneck.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        logger.info("The preprocessing will be online (recompute at each step)")

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
            logger.info(f"Starting {type(a)} augmentation...")
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
            dataset = dataset.get_all()
        for dataset in [a.aug_dataset for a in self._static_augmentations]:
            if type(dataset) == list:
                dataset = dataset + a.aug_dataset
        return dataset


class CachePreprocessor(AbstractPreprocessor):
    """
    Run the featurization before the start of the training and keep all feature in
    memory. Enabling fast training.
    Ideal when the featurization is time consuming but the final dataset fit in
    memory.
    """

    def __init__(self, config, use_disk: bool = False, disk_dir: str = "") -> None:
        """
        use_disk: if True, save the featurization to a cache list on the disk.
        disk_dir: where the cache list is saved
        """
        super().__init__(config)
        self._use_disk = use_disk
        if self._use_disk:
            logger.info("The preprocessing will be cache to DISK")
        else:
            logger.info("The preprocessing will be cache in the RAM")
        self._disk_dir = disk_dir if self._use_disk else None

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
        logger.info("Starting featurization...")
        dataset = self._run_parrallel(self.preprocess, loader, self._disk_dir)

        ids = [sample.id_ for sample in dataset]
        return ListLoader(dataset, ids)

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
        logger.info(f"Starting {type(a)} augmentation...")
        a.generate_augmented_data(loader)
        logger.info(f"Starting Featurization of  {type(a)} augmented data...")
        a.aug_dataset = self._run_parrallel(self.preprocess, a.aug_dataset)
        return a

    def _add_static_aug_dataset(self, dataset: ListLoader):
        tmp_dataset = dataset._dataset
        for a in self._static_augmentations:
            tmp_dataset += a.aug_dataset
        return ListLoader(tmp_dataset, range(len(tmp_dataset)))


class FilePreprocessor(AbstractPreprocessor):
    """
    This preprocessor is made to be used with the featurize task.
    It is a 2 step process, first run the featurization task with this preprocessor.
    Then use OnlinePreprocessor and load the generated feature with PickleLoadFeaturizer.
    The goal is to compute and save complex featurization. Ideal for large dataset with
    a time consuming featurization.
    If there is no need to access the featurization files it is also possible to use
    the Cached dataset with the `use_disk` option.
    """

    def __init__(
        self, config, folder_path: str, feature_to_save: List, input_to_use_has_filename: List, overwrite: bool = False
    ) -> None:
        """
        folder_path: Folder where the features will be save. Additional folder will be created
            based on the name of the feature to save.
        feature_to_save: Name after the featurization of which field to save.
            Will be used as additional folder name.
        input_to_use_has_filename: unique name for each file generated. This field
            can be used to skip the processing of identical feature and so speed up the preprocessing.
        """
        super().__init__(config)
        self.folder_name = Path(folder_path)
        self.feature_to_save = feature_to_save
        self.input_to_use_has_filename = input_to_use_has_filename
        self.overwrite = overwrite
        for input_field_filename in self.input_to_use_has_filename:
            folder = self.folder_name / input_field_filename
            folder.mkdir(exist_ok=True, parents=True)

    def _load_dataset(self) -> AbstractLoader:
        loader = SuperFactory.create(AbstractLoader, self._config.loader)
        with progress_bar() as progress:
            task_id = progress.add_task("File preprocessing:", total=len(loader))
            for n, sample in enumerate(loader):
                smiles = sample.inputs["smiles"] if "smiles" in sample.inputs else ""
                filenames = []
                for input_field_filename in self.input_to_use_has_filename:
                    filenames.append(self.folder_name / input_field_filename / f"{sample.inputs[input_field_filename]}.pkl")
                if not self.overwrite and all([filename.exists() for filename in filenames]):
                    continue
                try:
                    outputs = self.preprocess(sample)
                    for output_name, filename in zip(self.feature_to_save, filenames):
                        with open(filename, "wb") as file:
                            pickle.dump(outputs.inputs[output_name], file)
                except FeaturizationError as e:
                    logging.warning(e)
                except Exception as e:
                    logging.debug(f"{sample} {smiles} - {e}")
                progress.update(task_id, completed=n + 1, total=len(loader))

    def _load_augmented_data(self):
        return
