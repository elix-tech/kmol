from io import StringIO
from time import time, sleep
from pathlib import Path
from threading import Thread
from typing import Callable, Any

from kmol.core.logger import LOGGER as logging

from mila.configs import ClientConfiguration
from mila.exceptions import ClientAuthenticationError
from mila.services.box_utils import Box, File
from mila.services.clients.abstract_client import AbstractClient


class BoxClient(AbstractClient):
    def __init__(self, config: ClientConfiguration) -> None:
        super().__init__(config)
        self.box = Box(self._config.box_configuration)
        self.box.set_user(self._config.name)

    def _invoke(self, method: Callable, *args, **kwargs) -> Any:
        return method(*args, **kwargs)

    def _heartbeat_daemon(self) -> None:
        while True:
            self._invoke(self.heartbeat)
            sleep(self._config.heartbeat_frequency)

    def authenticate(self) -> File:
        """
        In Box our authentication goes with the creation of the heartbeat file.
        The server will use them to know the clients present during training.
        """
        auth_file_name = f"{self._config.name}_authentification.txt"
        auth_file = self.box.upload_text(self.box.auth_dir, auth_file_name, "")

        # Create first heartbeat file
        hb_file_name = f"{self._config.name}_heartbeat.txt"
        hb_file = self.box.upload_text(self.box.hb_dir, hb_file_name, str(time()))

        self.wait_for_authentification(auth_file)
        return hb_file

    def wait_for_authentification(self, file: File):
        while not self.connection_ended():
            content = file.content().decode()
            if content.split(" ")[0].lower() == "error":
                raise ClientAuthenticationError(content)
            elif content.split(" ")[0].lower() == "authenticated":
                return
            else:
                sleep(5)

    def heartbeat(self) -> None:
        stream = StringIO()
        stream.write(str(time()))
        stream.seek(0)
        self.hb_file = self.hb_file.update_contents_with_stream(stream)

    def close(self) -> None:
        pass

    def download_cfg(self):
        folder = self.box.get_folder_from_path(self.box.base_path)
        file = self.box.get_file_if_exist(folder, "cfg.json")
        # return bytes ?
        self.box.download_file(file, Path(self._config.save_path))
        with open(Path(self._config.save_path) / "cfg.json", "rb") as file:
            # Read the file's content in bytes
            return file.read()
        # return Path(self._config.save_path) / "cfg.json"

    def download_server_model(self, round_id) -> str:
        agg_file = info_file = None
        while agg_file is None and info_file is None:
            if self.connection_ended():
                return
            try:
                folder = self.box.get_folder_from_path(self.box.base_path / "rounds" / f"round_{round_id}")
                agg_file = self.box.get_file_if_exist(folder, "server.agg")
                if agg_file is None:
                    info_file = self.box.get_file_if_exist(folder, "no_starting_model.info")
            except:
                sleep(3)

        checkpoint_path = Path(self._config.save_path) / f"round_{round_id}" / "server.agg"
        if agg_file is not None:
            logging.info(f"Downloading aggregate model from the server for round {round_id}")
            self.box.download_file(agg_file, checkpoint_path.parent)
        else:
            logging.info(f"No starting point, strating the training")

        return checkpoint_path

    def upload_checkpoint(self, checkpoint_path: str, round_id) -> None:
        folder = self.box.get_folder_from_path(self.box.base_path / "rounds" / f"round_{round_id}")
        self.box.upload_file(f"{str(self._config.name).lower()}.pt", checkpoint_path, folder)

    def connection_ended(self):
        folder = self.box.get_folder_from_path(self.box.base_path)
        end_file = self.box.get_file_if_exist(folder, "server_ended.txt")
        if end_file is not None:
            logging.info("Server ended, ending the client")
        losing_connection_file = self.box.get_file_if_exist(folder, f"{self._config.name}.lost_connection")
        if losing_connection_file is not None:
            logging.info("Connection was lost for too long, the server cut the client from the training, ending the client")
        return end_file is not None or losing_connection_file is not None

    def run(self) -> None:
        try:
            self.hb_file = self._invoke(self.authenticate)
            heartbeat_worker = Thread(target=self._heartbeat_daemon)
            heartbeat_worker.daemon = True
            heartbeat_worker.start()

            configuration_content = self.download_cfg()

            round_id = 1
            while not self.connection_ended():
                agg_checkpoint_path = self._invoke(self.download_server_model, round_id=round_id)
                if self.connection_ended():
                    return
                configuration_path = self._create_configuration(configuration_content, agg_checkpoint_path)
                checkpoint_path = self._train(configuration_path)
                self._invoke(self.upload_checkpoint, checkpoint_path=checkpoint_path, round_id=round_id)
                round_id = round_id + 1

        except KeyboardInterrupt:
            logging.info("Stopping gracefully...")
            self._invoke(self.close)

        except Exception as e:
            logging.error("[internal error] {}".format(e))
            self._invoke(self.close)
