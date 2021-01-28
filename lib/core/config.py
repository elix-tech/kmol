import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any

import torch

from mila.factories import AbstractConfiguration


@dataclass
class Config(AbstractConfiguration):

    model: Dict[str, Any]
    loader: Dict[str, Any]
    splitter: Dict[str, Any]
    featurizers: List[Dict[str, Any]]

    checkpoint_path: Optional[str] = None
    threshold: float = 0.5

    train_split: str = "train"
    test_split: str = "test"

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0

    use_cuda: bool = True
    enabled_gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    cache_location: str = "/tmp/federated/"
    clear_cache: bool = True

    log_level: Literal["debug", "info", "warn", "error", "critical"] = "info"
    log_format: str = ""
    log_frequency: int = 20

    def should_parallelize(self) -> bool:
        return torch.cuda.is_available() and self.use_cuda and len(self.enabled_gpus) > 1

    def get_device(self) -> torch.device:
        device_id = self.enabled_gpus[0] if len(self.enabled_gpus) == 1 else 0
        device_name = "cuda:" + str(device_id) if torch.cuda.is_available() and self.use_cuda else "cpu"

        return torch.device(device_name)

    def _after_load(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        logging.basicConfig(format=self.log_format, level=self.log_level.upper())
