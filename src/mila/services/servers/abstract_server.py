
from ...configs import ServerConfiguration
from ..server_manager.server_manager import IOManager, ServerManager

class AbstractServer(IOManager):

    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config

    def run(self, servicer: ServerManager) -> None:
        raise NotImplementedError()

