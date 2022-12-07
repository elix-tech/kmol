
from threading import Thread
from kmol.core.logger import LOGGER as logging
from time import sleep
from pathlib import Path

from ...configs import ServerConfiguration
from ..server_manager.box_servicer import BoxServicer
from .abstract_server import AbstractServer
from ..clients.box_utils import Box, File


class BoxServer(AbstractServer):

    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config
        self.bbox = Box(self._config.bbox_configuration)

    
    def _daemon(self, func) -> None:
        while True:

            self._invoke(func)
            sleep(self.bbox_cfg.scan_frequency)


        

    def run(self, servicer: BoxServicer) -> None:

        authentication_worker = Thread(target=self._daemon, args=(servicer._authenticate))
        authentication_worker.daemon = True
        authentication_worker.start()

        heartbeat_worker = Thread(target=self._daemon, args=(servicer._heartbeat))
        heartbeat_worker.daemon = True
        heartbeat_worker.start()



