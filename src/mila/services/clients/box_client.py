import json
from io import StringIO
from time import time, sleep
from pathlib import Path
from threading import Thread
from typing import Dict, Callable, Any, Type, Optional

from boxsdk import Client

from kmol.core.logger import LOGGER as logging

from .abstract_client import AbstractClient
from ...factories import ClientConfiguration
from .box_utils import Box, File
from ...exceptions import ClientAuthenticationError

class BoxClient(AbstractClient):

    def __init__(self, config: ClientConfiguration) -> None:
        super().__init__(config)
        self._cfg_bbox = self._config.bbox_configuration
        self.client = self._connect()
        self.user_client = self._get_user_client(self.client)
        self.hb_file = self.authenticate()
        self.bbox = Bbox()
        self.bbox.set_user(self._config.name)


    def _invoke(self, method: Callable, *args, **kwargs) -> Any:
        return method(*args, **kwargs)

    def _heartbeat_daemon(self) -> None:
        while True:

            if not self._token:
                break

            self._invoke(self.heartbeat)
            sleep(self._config.heartbeat_frequency)

    def authenticate(self) -> File:
        """
        In Box our authentication goes with the creation of the heartbeat file.
        The server will use them to know the clients present during training.
        """
        auth_file_name = f"{self._config.name}_authentification.txt"
        auth_file = self.bbox.upload_text(self.bbox.hb_dir, auth_file_name, "")
        
        # Create first heartbeat file
        hb_file_name = f"{self._config.name}_heartbeat.txt"
        hb_file = self.bbox.upload_text(self.bbox.hb_dir, hb_file_name, str(time()))

        self.wait_for_authentification(auth_file)
        return hb_file
    
    def wait_for_authentification(self, file: File):
        while True:
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

    def request_model(self) -> str:
        pass
        
    def send_checkpoint(self, checkpoint_path: str) -> None:
        with open(checkpoint_path, "rb") as read_buffer:
            content = read_buffer.read()

    def run(self) -> None:
        try:
            self._token = self._invoke(self.authenticate)

            heartbeat_worker = Thread(target=self._heartbeat_daemon)
            heartbeat_worker.daemon = True
            heartbeat_worker.start()

            while True:
                configuration_path = self._invoke(self.request_model)
                checkpoint_path = self._train(configuration_path)
                self._invoke(self.send_checkpoint, checkpoint_path=checkpoint_path)

        except KeyboardInterrupt:
            logging.info("Stopping gracefully...")
            self._invoke(self.close)

        except Exception as e:
            logging.error("[internal error] {}".format(e))
            self._invoke(self.close)







            

    # def _connect(self) -> Client:
    #     auth = JWTAuth.from_settings_file(self._cfg_bbox.box_configuration_path)
    #     return Client(auth)
    
    # def _get_user_client(self, client: Client) -> Client:
    #     users = get_users_with_names(client, self._config.name)
    #     if len(users) > 0:
    #         user = users[0]
    #     else:
    #         user = client.create_user(self._config.name, login=None)
    #     return client.as_user(user)

# def _invoke(self, method: Callable, *args, **kwargs) -> Any:
#         return method(*args, **kwargs)

#     def _heartbeat_daemon(self) -> None:
#         while True:

#             if not self._token:
#                 break

#             self._invoke(self.heartbeat)
#             sleep(self._config.heartbeat_frequency)

#     def authenticate(self) -> File:
#         """
#         In Box our authentication goes with the creation of the heartbeat file.
#         The server will use them to know the clients present during training.
#         """
#         auth_file_name = f"{self._config.name}_authentification.txt"
#         auth_path = Path(self._cfg_bbox.shared_dir_name) / self._cfg_bbox.save_path / 'authentifications'
#         auth_folder = get_folder_from_path(self.user_client, auth_path)
#         auth_file = upload_text(auth_folder, auth_file_name, "")
        
#         # Create first heartbeat file
#         hb_file_name = f"{self._config.name}_heartbeat.txt"
#         hb_path = Path(self._cfg_bbox.shared_dir_name) / self._cfg_bbox.save_path / 'heartbeats'
#         hb_folder = get_folder_from_path(self.user_client, hb_path)
#         hb_file = upload_text(hb_folder, hb_file_name, str(time()))

#         self.wait_for_authentification(auth_file)
#         return hb_file
    
#     def wait_for_authentification(self, file: File):
#         while True:
#             content = file.content().decode()
#             if content.split(" ")[0].lower() == "error":
#                 raise ClientAuthenticationError(content)
#             elif content.split(" ")[0].lower() == "authenticated":
#                 return
#             else:
#                 sleep(5)

#     def heartbeat(self) -> None:
#         stream = StringIO()
#         stream.write(str(time()))
#         stream.seek(0)
#         self.hb_file = self.hb_file.update_contents_with_stream(stream)

#     def close(self) -> None:
#         pass

#     def request_model(self) -> str:
#         pass
        
#     def send_checkpoint(self, checkpoint_path: str) -> None:
#         with open(checkpoint_path, "rb") as read_buffer:
#             content = read_buffer.read()

#     def run(self) -> None:
#         try:
#             self._token = self._invoke(self.authenticate)

#             heartbeat_worker = Thread(target=self._heartbeat_daemon)
#             heartbeat_worker.daemon = True
#             heartbeat_worker.start()

#             while True:
#                 configuration_path = self._invoke(self.request_model)
#                 checkpoint_path = self._train(configuration_path)
#                 self._invoke(self.send_checkpoint, checkpoint_path=checkpoint_path)

#         except KeyboardInterrupt:
#             logging.info("Stopping gracefully...")
#             self._invoke(self.close)

#         except Exception as e:
#             logging.error("[internal error] {}".format(e))
#             self._invoke(self.close)