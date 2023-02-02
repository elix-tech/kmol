import re
from threading import Thread
from time import sleep
from typing import Callable, Any

import grpc
from kmol.core.logger import LOGGER as logging

from ...protocol_buffers import mila_pb2, mila_pb2_grpc
from ...exceptions import InvalidNameError
from .abstract_client import AbstractClient
from ...configs import ClientConfiguration

class GrcpClient(AbstractClient):

    def __init__(self, config: ClientConfiguration) -> None:
        super().__init__(config)
        self._cfg_grcp = self._config.grcp_configuration

        self._validate()

    def _validate(self) -> None:
        if not re.match(r"^[\w]+$", self._config.name):
            raise InvalidNameError("[ERROR] The client name can only contain alphanumeric characters and underscores.")

    def _invoke(self, method: Callable, *args, **kwargs) -> Any:
        with self._connect() as channel:
            while True:
                try:
                    stub = mila_pb2_grpc.MilaStub(channel)
                    kwargs["stub"] = stub

                    return method(*args, **kwargs)

                except grpc.RpcError as e:
                    if grpc.StatusCode.RESOURCE_EXHAUSTED == e.code():
                        sleep(self._config.retry_timeout)
                        continue

                    raise e
    
    def _connect(self) -> grpc.Channel:
        if self._cfg_grcp.use_secure_connection:
            credentials = self._get_credentials()
            return grpc.secure_channel(target=self._cfg_grcp.target, credentials=credentials, options=self._cfg_grcp.options)
        else:
            logging.warning("[CAUTION] Connection is insecure!")
            return grpc.insecure_channel(self._cfg_grcp.target, options=self._cfg_grcp.options)

    def _get_credentials(self) -> grpc.ServerCredentials:
        private_key = self._read_file(self._cfg_grcp.ssl_private_key)
        certificate_chain = self._read_file(self._cfg_grcp.ssl_cert)
        root_certificate = self._read_file(self._cfg_grcp.ssl_root_cert)

        return grpc.ssl_channel_credentials(
            certificate_chain=certificate_chain, private_key=private_key, root_certificates=root_certificate
        )

    def _heartbeat_daemon(self) -> None:
        while True:

            if not self._token:
                break

            self._invoke(self.heartbeat)
            sleep(self._config.heartbeat_frequency)

    def authenticate(self, stub: mila_pb2_grpc.MilaStub) -> str:
        package = mila_pb2.Client(name=self._config.name)
        response = stub.Authenticate(package)

        return response.token

    def heartbeat(self, stub: mila_pb2_grpc.MilaStub) -> None:
        package = mila_pb2.Token(token=self._token)
        stub.Heartbeat(package)

    def close(self, stub: mila_pb2_grpc.MilaStub) -> None:
        package = mila_pb2.Token(token=self._token)
        stub.Close(package)

        self._token = None

    def request_model(self, stub: mila_pb2_grpc.MilaStub) -> str:
        package = mila_pb2.Token(token=self._token)
        response = stub.RequestModel(package)

        checkpoint_path = None
        if response.latest_checkpoint:
            checkpoint_path = self._store_checkpoint(response.latest_checkpoint)

        return self._create_configuration(response.json_configuration, checkpoint_path)

    def _store_checkpoint(self, checkpoint: bytes) -> str:
        checkpoint_path = "{}/checkpoint.latest".format(self._config.save_path)
        with open(checkpoint_path, "wb") as write_buffer:
            write_buffer.write(checkpoint)

        return checkpoint_path


    def send_checkpoint(self, checkpoint_path: str, stub: mila_pb2_grpc.MilaStub) -> None:
        with open(checkpoint_path, "rb") as read_buffer:
            content = read_buffer.read()

        package = mila_pb2.Checkpoint(token=self._token, content=content)
        stub.SendCheckpoint(package)

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

        except grpc.RpcError as e:
            logging.info("[{}] {}".format(e.code(), e.details()))
            if e.code() not in (grpc.StatusCode.PERMISSION_DENIED, grpc.StatusCode.UNAVAILABLE):
                self._invoke(self.close)

        except KeyboardInterrupt:
            logging.info("Stopping gracefully...")
            self._invoke(self.close)

        except Exception as e:
            logging.error("[internal error] {}".format(e))
            self._invoke(self.close)
