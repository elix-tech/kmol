import importlib
from typing import Callable
class IOManager:

    def _read_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as read_buffer:
            return read_buffer.read()

    @classmethod
    def _reflect(cls, object_path: str) -> Callable:
        module, class_name = object_path.rsplit(".", 1)
        return getattr(importlib.import_module(module), class_name)