from abc import abstractmethod
from typing import Any, Dict

import torch
from ...model.architectures import AbstractNetwork


class AbstractCaptumNetwork(AbstractNetwork):
    @abstractmethod
    def get_integrate_gradient(self, data: Dict[str, Any], **kwargs) -> torch.Tensor:
        raise NotImplementedError
