import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, DefaultDict

import torch
from mila.factories import AbstractConfiguration

from .helpers import SuperFactory
from .observers import AbstractEventHandler, EventManager, DifferentialPrivacy


@dataclass
class Config(AbstractConfiguration):

    model: Dict[str, Any]
    loader: Dict[str, Any]
    splitter: Dict[str, Any]
    featurizers: List[Dict[str, Any]]
    transformers: List[Dict[str, Any]]
    criterion: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]

    is_stepwise_scheduler: Optional[bool] = True
    is_finetuning: Optional[bool] = False
    checkpoint_path: Optional[str] = None
    threshold: Optional[float] = None
    inference_mode: Optional[str] = None
    cross_validation_folds: int = 5
    mc_dropout_iterations: int = 5
    mc_dropout_probability: Optional[float] = None
    probe_layer: Optional[str] = None

    train_split: str = "train"
    train_metrics: List[str] = field(default_factory=lambda: [])
    validation_split: str = "validation"
    test_split: str = "test"
    test_metrics: List[str] = field(default_factory=lambda: [])

    epochs: int = 100
    batch_size: int = 32
    drop_last_batch: bool = False

    use_cuda: bool = True
    enabled_gpus: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 0
    featurization_jobs: int = 4

    cache_location: str = "/tmp/federated/"
    clear_cache: bool = False

    log_level: Literal["debug", "info", "warn", "error", "critical"] = "info"
    log_format: str = ""
    log_frequency: int = 20
    overwrite_checkpoint: bool = False

    observers: DefaultDict[str, List[Dict]] = field(default_factory=lambda: defaultdict(list))
    differential_privacy: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})

    target_metric: str = "roc_auc"
    optuna_trials: int = 1000
    optuna_init: Optional[Dict[str, Any]] = None
    subset: Optional[Dict[str, Any]] = None
    visualizer: Optional[Dict[str, Any]] = None

    def should_parallelize(self) -> bool:
        return torch.cuda.is_available() and self.use_cuda and len(self.enabled_gpus) > 1

    def get_device(self) -> torch.device:
        device_id = self.enabled_gpus[0] if len(self.enabled_gpus) == 1 else 0
        device_name = "cuda:" + str(device_id) if torch.cuda.is_available() and self.use_cuda else "cpu"

        return torch.device(device_name)

    def __post_init__(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        with open(Path(self.output_path) / "config.json", 'w') as file:
            json.dump(self.__dict__, file, indent = 2)
        
        logging.basicConfig(format=self.log_format, level=self.log_level.upper())

        EventManager.flush()
        for event_name, event_handlers in self.observers.items():
            for event_handler_definition in event_handlers:
                event_handler = SuperFactory.create(AbstractEventHandler, event_handler_definition)
                EventManager.add_event_listener(event_name=event_name, handler=event_handler)

        if self.differential_privacy["enabled"]:
            DifferentialPrivacy.setup(**self.differential_privacy["options"])

    def cloned_update(self, **kwargs) -> "Config":
        options = deepcopy(vars(self))
        options.update(**kwargs)

        return Config(**options)
