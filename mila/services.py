import json
import logging
import os
import re
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from glob import glob
from threading import Thread
from time import time, sleep
from typing import Dict, Callable, Any

import grpc

from lib.config import Config
from mila.configs import ServerConfiguration, ClientConfiguration
from mila.exceptions import InvalidNameError
from mila.protocol_buffers import mila_pb2, mila_pb2_grpc
from run import Executor


class ByteReader:

    def _read_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as read_buffer:
            return read_buffer.read()


@dataclass
class Participant:
    name: str
    ip_address: str
    token: str = field(default_factory=uuid.uuid4)
    __last_heartbeat: float = field(default_factory=time)

    def register_heartbeat(self) -> None:
        self.__last_heartbeat = time()

    def is_alive(self, heartbeat_timeout: float) -> bool:
        return time() - self.__last_heartbeat < heartbeat_timeout

    def __eq__(self, other: "Participant") -> bool:
        return self.name == other.name and self.ip_address == other.ip_address

    def __str__(self) -> str:
        return "{}|{}".format(self.name, self.ip_address)


class ServerManager(ByteReader):

    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config
        self._registry: Dict[str, Participant] = {}

        self._current_round = 0
        self._latest_checkpoint = None

    def verify_ip(self, ip_address: str) -> bool:
        if ip_address in self._config.blacklist:
            return False

        if self._config.use_whitelist and ip_address not in self._config.whitelist:
            return False

        return True

    def register_client(self, name: str, ip_address: str) -> mila_pb2.Token:
        client = Participant(name=name, ip_address=ip_address)

        for entry in self._registry.values():
            if client == entry:
                return entry.token

        self._registry[client.token] = client
        logging.info("[{}] Successfully authenticated (clients={})".format(client, len(self._registry)))

        return client.token

    def verify_token(self, token: str, ip_address: str) -> bool:
        return (
            token in self._registry
            and self._registry[token].ip_address == ip_address
            and self._registry[token].is_alive(self._config.heartbeat_timeout)
        )

    def register_heartbeat(self, token: str) -> None:
        self._registry[token].register_heartbeat()

    def close_connection(self, token: str) -> None:
        client = self._registry[token]
        logging.info("[{}] Disconnected (clients={})".format(client, len(self._registry)))

        self._registry.pop(token)

    def save_checkpoint(self, token: str, content: bytes) -> None:
        client = self._registry[token]
        save_path = "{}/{}.{}.{}.remote".format(
            self._config.save_path, client.name, client.ip_address.replace(".", "_"), self._current_round
        )

        with open(save_path, "wb") as write_buffer:
            write_buffer.write(content)

        logging.info("[{}] Checkpoint Received".format(client))

    def get_configuration(self) -> bytes:
        return self._read_file(self._config.task_configuration_file)

    def get_latest_checkpoint(self) -> bytes:
        if self._latest_checkpoint is None:
            return b""

        return self._read_file(self._latest_checkpoint)


class DefaultServicer(ServerManager, mila_pb2_grpc.MilaServicer):

    def _validate_token(self, token: str, context) -> bool:
        success = self.verify_token(token=token, ip_address=self._get_ip(context))

        if not success:
            context.set_details("Access Denied... Token is invalid...")
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)

        return success
    
    def _get_ip(self, context) -> str:
        return context.peer().split(':')[1]

    def Authenticate(self, request: grpc, context) -> str:
        client_ip = self._get_ip(context)

        if not self.verify_ip(client_ip):
            context.set_details("Access Denied... IP Address is not whitelisted.")
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)

            return mila_pb2.Token()

        return self.register_client(request.client_name, client_ip)

    def Heartbeat(self, request, context) -> None:
        if self._validate_token(request.token, context):
            self.register_heartbeat(request.token)

    def Close(self, request, context) -> None:
        if self._validate_token(request.token, context):
            self.close_connection(request.token)

    def RequestModel(self, request, context) -> mila_pb2.Model:
        if self._validate_token(request.token, context):
            return mila_pb2.Model(
                configuration=self.get_configuration(),
                latest_checkpoint=self.get_latest_checkpoint()
            )

    def SendCheckpoint(self, request, context) -> None:
        if self._validate_token(request.token, context):
            self.save_checkpoint(token=request.token, content=request.content)


class Server(ByteReader):

    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config

    def _get_credentials(self) -> grpc.ServerCredentials:
        private_key = self._read_file(self._config.ssl_private_key)
        certificate_chain = self._read_file(self._config.ssl_cert)
        root_certificate = self._read_file(self._config.ssl_root_cert)

        return grpc.ssl_server_credentials(
            ((private_key, certificate_chain),),
            root_certificates=root_certificate,
            require_client_auth=True
        )

    def run(self, servicer: mila_pb2_grpc.MilaServicer) -> None:
        workers_count = max(self._config.workers, self._config.minimum_clients)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers_count), options=self._config.options)
        mila_pb2_grpc.add_MilaServicer_to_server(servicer, server)

        if self._config.use_secure_connection:
            credentials = self._get_credentials()
            server.add_secure_port(self._config.target, credentials)
        else:
            server.add_insecure_port(self._config.target)
            logging.warning("[CAUTION] Connection is insecure!")

        logging.info("Starting server at: [%s]".format(self._config.target))

        server.start()
        server.wait_for_termination()


class Client(ByteReader):

    def __init__(self, config: ClientConfiguration) -> None:
        self._config = config
        self._token = None

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
            return grpc.secure_channel(target=self._config.target, credentials=credentials)
        else:
            logging.warning("[CAUTION] Connection is insecure!")
            return grpc.insecure_channel(self._config.target)

    def _store_checkpoint(self, checkpoint: bytes) -> str:
        checkpoint_path = "{}/checkpoint.latest".format(self._config.save_path)
        with open(checkpoint_path, "wb") as write_buffer:
            write_buffer.write(checkpoint)

        return checkpoint_path

    def _create_configuration(self, received_configuration: bytes, checkpoint_path: str) -> str:
        configuration = received_configuration.decode("utf-8")

        configuration = {**configuration, **self._config.model_overwrites}  # overwrite values based on settings
        configuration["checkpoint_path"] = checkpoint_path

        configuration_path = "{}/checkpoint.latest".format(self._config.save_path)
        with open(configuration_path, "w") as write_buffer:
            json.dump(configuration, write_buffer)

        return configuration_path

    def _retrieve_latest_file(self, folder_path: str) -> str:
        files = glob("{}/*".format(folder_path))
        return max(files, key=os.path.getctime)

    # TODO: Mila should not depend on other modules
    #       - move the config and a base Executor to Mila (lib should extend from them);
    #       - find a way to add abstraction to NamedTuples (maybe use a dataclass instead?)
    def _train(self, configuration_path: str) -> str:
        config = Config.load(configuration_path)

        runner = Executor(config=config)
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
                        sleep(5)
                    else:
                        self._token = None
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

        checkpoint_path = self._store_checkpoint(response.latest_checkpoint)
        return self._create_configuration(response.json_configuration, checkpoint_path)

    def send_checkpoint(self, checkpoint_path: str, stub: mila_pb2_grpc.MilaStub) -> None:
        with open(checkpoint_path, "rb") as read_buffer:
            content = read_buffer.read()

        package = mila_pb2.Checkpoint(token=self._token, content=content)
        stub.SendCheckpoint(package)

    def run(self) -> None:
        self._token = self._invoke(self.authenticate)

        heartbeat_worker = Thread(target=self._heartbeat_daemon)
        heartbeat_worker.daemon = True
        heartbeat_worker.start()

        while True:
            try:
                configuration_path = self._invoke(self.request_model)
                checkpoint_path = self._train(configuration_path)
                self._invoke(self.send_checkpoint, checkpoint_path=checkpoint_path)
            except KeyboardInterrupt:
                self._invoke(self.close)


# TODO: implement server handlers (wait for minimum number of clients, etc.)
