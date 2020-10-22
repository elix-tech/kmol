import json
import logging
import os
from typing import NamedTuple, Literal, Optional, List, Dict, Any, Iterable

import torch


class Config(NamedTuple):

    model_name: Literal["GraphConvolutionalNetwork", "GraphIsomorphismNetwork"]
    model_options: Dict[str, Any]

    data_loader: Literal["MoleculeNetLoader"]
    dataset: Literal["tox21", "pcba", "muv", "hiv", "bbbp", "toxcast", "sider", "clintox"]

    input_path: str
    output_path: str
    checkpoint_path: Optional[str] = None

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0
    dropout: float = 0
    hidden_layer_size: int = 16

    threshold: float = 0.5
    train_ratio: float = 0.8
    split_method: Literal["index", "random"] = "index"
    seed: int = 42

    use_cuda: bool = True
    enabled_gpus: List[int] = [0, 1, 2, 3]

    log_level: Literal["debug", "info", "warn", "error", "critical"] = "info"
    log_format: str = ""
    log_frequency: int = 20

    def get_data_loader(self, mode=Literal["train", "test"]) -> Iterable:
        from lib import data_loaders

        data_loader = getattr(data_loaders, self.data_loader)
        return data_loader(config=self, mode=mode)

    def get_model(self) -> torch.nn.Module:
        from lib import models

        model = getattr(models, self.model_name)
        return model(**self.model_options)

    def get_device(self) -> torch.device:
        device_id = self.enabled_gpus[0] if len(self.enabled_gpus) == 1 else 0
        device_name = "cuda:" + str(device_id) if torch.cuda.is_available() and self.use_cuda else "cpu"

        return torch.device(device_name)

    @classmethod
    def load(cls, file_path: str) -> "Config":
        with open(file_path) as read_handle:
            config = cls(**json.load(read_handle))

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        logging.basicConfig(format=config.log_format, level=config.log_level.upper())
        return config
