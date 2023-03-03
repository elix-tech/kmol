import json
import os
from glob import glob
from threading import Thread
from time import time, sleep
from typing import Dict, Callable, Any, Type, Optional

from kmol.core.logger import LOGGER as logging

from ...configs import ClientConfiguration
from ...factories import AbstractConfiguration, AbstractExecutor
from ...services import IOManager


class AbstractClient(IOManager):
    
    def __init__(self, config: ClientConfiguration) -> None:
        self._config = config
        self._token = None

        if not os.path.exists(self._config.save_path):
            os.makedirs(self._config.save_path)

    def _create_configuration(self, received_configuration: bytes, checkpoint_path: Optional[str]) -> str:
        configuration = json.loads(received_configuration.decode("utf-8"))

        configuration = {**configuration, **self._config.model_overwrites}  # overwrite values based on settings
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            configuration["checkpoint_path"] = str(checkpoint_path)

        configuration_path = "{}/config.latest.json".format(self._config.save_path)
        with open(configuration_path, "w") as write_buffer:
            json.dump(configuration, write_buffer)

        return configuration_path

    def _retrieve_latest_file(self, folder_path: str) -> str:
        files = glob("{}/*.pt".format(folder_path))
        return max(files, key=os.path.getctime)

    def _train(self, configuration_path: str) -> str:
        config: Type[AbstractConfiguration] = self._reflect(self._config.config_type)
        config = config.from_file(configuration_path, "train")

        runner: Type[AbstractExecutor] = self._reflect(self._config.executor_type)
        runner = runner(config=config)
        runner.train()

        return self._retrieve_latest_file(config.output_path)

    def _invoke(self, method: Callable, *args, **kwargs) -> Any:
        return method(*args, **kwargs)

    def _heartbeat_daemon(self) -> None:
        while True:

            if not self._token:
                break

            self._invoke(self.heartbeat)
            sleep(self._config.heartbeat_frequency)

    def authenticate(self) -> str:
        pass

    def heartbeat(self) -> None:
        pass

    def close(self) -> None:
        pass

    def request_model(self) -> str:
        pass
        
    def send_checkpoint(self, checkpoint_path: str) -> None:
        pass

    def run(self) -> None:
        try:
            self._token = self._invoke(self.authenticate)

            heartbeat_worker = Thread(target=self._heartbeat_daemon)
            heartbeat_worker.daemon = True
            heartbeat_worker.start()

            # while True:
            #     configuration_path = self._invoke(self.request_model)
            #     checkpoint_path = self._train(configuration_path)
            #     self._invoke(self.send_checkpoint, checkpoint_path=checkpoint_path)

        except KeyboardInterrupt:
            logging.info("Stopping gracefully...")
            self._invoke(self.close)

        except Exception as e:
            logging.error("[internal error] {}".format(e))
            self._invoke(self.close)
