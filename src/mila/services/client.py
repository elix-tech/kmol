import json
import os
import re
from glob import glob
from threading import Thread
from time import time, sleep
from typing import Dict, Callable, Any, Type, Optional

import grpc
from kmol.core.logger import LOGGER as logging

from ..configs import ClientConfiguration
from ..exceptions import InvalidNameError
from ..factories import AbstractConfiguration, AbstractExecutor
from ..protocol_buffers import mila_pb2, mila_pb2_grpc


class Client(IOManager):

    def __init__(self, config: ClientConfiguration) -> None:
        self._config = config
        self._token = None

        if not os.path.exists(self._config.save_path):
            os.makedirs(self._config.save_path)

        self._validate()

    def _validate(self) -> None:
        if not re.match(r"^[\w]+$", self._config.name):
            raise InvalidNameError(
                "[ERROR] The client name can only contain alphanumeric characters and underscores."
            )

    def _get_credentials(self) -> grpc.ServerCredentials:
        private_key = self._read_file(self._config.ssl_private_key)
        certificate_chain = self._read_file(self._config.ssl_cert)
        root_certificate = self._read_file(self._config.ssl_root_cert)

        return grpc.ssl_channel_credentials(
            certificate_chain=certificate_chain,
            private_key=private_key,
            root_certificates=root_certificate
        )

    def _connect(self) -> grpc.Channel:
        if self._config.use_secure_connection:
            credentials = self._get_credentials()
            return grpc.secure_channel(
                target=self._config.target, credentials=credentials, options=self._config.options
            )
        else:
            logging.warning("[CAUTION] Connection is insecure!")
            return grpc.insecure_channel(self._config.target, options=self._config.options)

    def _store_checkpoint(self, checkpoint: bytes) -> str:
        checkpoint_path = "{}/checkpoint.latest".format(self._config.save_path)
        with open(checkpoint_path, "wb") as write_buffer:
            write_buffer.write(checkpoint)

        return checkpoint_path

    def _create_configuration(self, received_configuration: bytes, checkpoint_path: Optional[str]) -> str:
        configuration = json.loads(received_configuration.decode("utf-8"))

        configuration = {**configuration, **self._config.model_overwrites}  # overwrite values based on settings
        configuration["checkpoint_path"] = checkpoint_path

        configuration_path = "{}/config.latest.json".format(self._config.save_path)
        with open(configuration_path, "w") as write_buffer:
            json.dump(configuration, write_buffer)

        return configuration_path

    def _retrieve_latest_file(self, folder_path: str) -> str:
        files = glob("{}/*.pt".format(folder_path))
        return max(files, key=os.path.getctime)

    def _train(self, configuration_path: str) -> str:
        config: Type[AbstractConfiguration] = self._reflect(self._config.config_type)
        config = config.from_file(configuration_path)

        runner: Type[AbstractExecutor] = self._reflect(self._config.executor_type)
        runner = runner(config=config)
        runner.train()

        return self._retrieve_latest_file(config.output_path)

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
