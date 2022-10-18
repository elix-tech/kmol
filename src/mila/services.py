import importlib
import json
import os
import re
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from glob import glob
from threading import Thread, Lock
from time import time, sleep
from typing import Dict, Callable, Any, Type, Optional

import grpc
from google.protobuf.empty_pb2 import Empty as EmptyResponse
from kmol.core.logger import LOGGER as logging

from .configs import ServerConfiguration, ClientConfiguration, LocalConfiguration
from .exceptions import InvalidNameError, ClientAuthenticationError
from .factories import AbstractConfiguration, AbstractExecutor, AbstractAggregator
from .protocol_buffers import mila_pb2, mila_pb2_grpc


class IOManager:

    def _read_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as read_buffer:
            return read_buffer.read()

    def _reflect(self, object_path: str) -> Callable:
        module, class_name = object_path.rsplit(".", 1)
        return getattr(importlib.import_module(module), class_name)


@dataclass
class Participant:
    name: str
    ip_address: str

    token: str = field(default_factory=lambda: str(uuid.uuid4()))
    round: int = 0
    awaiting_response: bool = False

    __last_heartbeat: float = field(default_factory=time)

    def register_heartbeat(self) -> None:
        self.__last_heartbeat = time()

    def is_alive(self, heartbeat_timeout: float) -> bool:
        return time() - self.__last_heartbeat < heartbeat_timeout

    def __eq__(self, other: "Participant") -> bool:
        return self.name == other.name and self.ip_address == other.ip_address

    def __str__(self) -> str:
        return "{}|{}".format(self.name, self.ip_address)


class ServerManager(IOManager):

    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config
        self._registry: Dict[str, Participant] = {}

        self._current_round = 1
        self._latest_checkpoint = self._config.start_point
        self._last_registration_time = 0
        self._is_registration_closed = False

        if not os.path.exists(self._config.save_path):
            os.makedirs(self._config.save_path)

    def verify_ip(self, ip_address: str) -> bool:
        if ip_address in self._config.blacklist:
            return False

        if self._config.use_whitelist and ip_address not in self._config.whitelist:
            return False

        return True

    def register_client(self, name: str, ip_address: str) -> str:
        client = Participant(name=name, ip_address=ip_address)
        for entry in self._registry.values():
            if client == entry:
                return entry.token

        if not self.should_wait_for_additional_clients():
            raise ClientAuthenticationError("Authentication failed... Registration is closed.")

        self._registry[client.token] = client
        self._last_registration_time = time()

        logging.info("[{}] Successfully authenticated (clients={})".format(client, self.get_clients_count()))
        return client.token

    def verify_token(self, token: str, ip_address: str) -> bool:
        return (
            token in self._registry
            and self._registry[token].ip_address == ip_address
            and self._registry[token].is_alive(self._config.heartbeat_timeout)
        )

    def register_heartbeat(self, token: str) -> None:
        client = self._registry[token]

        client.register_heartbeat()
        logging.debug("[{}] Heartbeat registered".format(client))

    def close_connection(self, token: str) -> None:
        client = self._registry[token]

        self._registry.pop(token)
        logging.info("[{}] Disconnected (clients={})".format(client, self.get_clients_count()))

    def save_checkpoint(self, token: str, content: bytes) -> None:
        client = self._registry[token]
        save_path = self.get_client_filename_for_current_round(client)

        with open(save_path, "wb") as write_buffer:
            write_buffer.write(content)

        logging.info("[{}] Checkpoint Received".format(client))

    def get_configuration(self) -> bytes:
        return self._read_file(self._config.task_configuration_file)

    def get_latest_checkpoint(self) -> bytes:
        if self._latest_checkpoint is None:
            return b""

        return self._read_file(self._latest_checkpoint)

    def close_registration(self) -> None:
        self._is_registration_closed = True

    def should_wait_for_additional_clients(self) -> bool:
        if self._is_registration_closed:
            return False

        clients_count = self.get_clients_count()

        return (
            clients_count < self._config.minimum_clients
            or (
                clients_count < self._config.maximum_clients
                and time() - self._last_registration_time < self._config.client_wait_time
            )
        )

    def are_more_rounds_required(self) -> bool:
        return self._current_round <= self._config.rounds_count

    def set_client_status_to_awaiting_response(self, token: str) -> bool:
        client = self._registry[token]
        if client.round >= self._current_round:
            return False

        client.round = self._current_round
        client.awaiting_response = True
        return True

    def set_client_status_to_available(self, token: str) -> None:
        self._registry[token].awaiting_response = False

    def are_all_updates_received(self) -> bool:
        for client in self._registry.values():
            if client.awaiting_response or client.round != self._current_round:
                return False

        return True

    def aggregate(self) -> None:
        logging.info("Start aggregation (round={})".format(self._current_round))

        checkpoint_paths = self.get_clients_model_path_for_current_round()
        save_path = "{}/{}.aggregate.pt".format(self._config.save_path, self._current_round)

        aggregator: Type[AbstractAggregator] = self._reflect(self._config.aggregator_type)
        aggregator(**self._config.aggregator_options).run(checkpoint_paths=checkpoint_paths, save_path=save_path)

        logging.info("Aggregate model saved: [{}]".format(save_path))
        self._latest_checkpoint = save_path

    def enable_next_round(self) -> None:
        self._current_round += 1
        if self._current_round <= self._config.rounds_count:
            logging.info("Starting round [{}]".format(self._current_round))

    def get_clients_count(self) -> int:
        return sum(1 for client in self._registry.values() if client.is_alive(self._config.heartbeat_timeout))

    def get_clients_model_path_for_current_round(self):
        return [
            self.get_client_filename_for_current_round(client)
            for client in self._registry.values()
        ]

    def get_client_filename_for_current_round(self, client: Participant):
        return "{}/{}.{}.{}.remote".format(
            self._config.save_path,
            client.name,
            client.ip_address.replace(".", "_"),
            self._current_round
        )


class DefaultServicer(ServerManager, mila_pb2_grpc.MilaServicer):

    def __init__(self, config: ServerConfiguration) -> None:
        super().__init__(config=config)
        self.__lock = Lock()

    def _validate_token(self, token: str, context) -> bool:
        if not self.verify_token(token=token, ip_address=self._get_ip(context)):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access Denied... Token is invalid...")

        return True

    def _get_ip(self, context) -> str:
        return context.peer().split(':')[1]

    def Authenticate(self, request: grpc, context) -> str:
        client_ip = self._get_ip(context)
        if not self.verify_ip(client_ip):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access Denied... IP Address is not whitelisted.")

        try:
            token = self.register_client(request.name, client_ip)
            return mila_pb2.Token(token=token)
        except ClientAuthenticationError as e:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))

    def Heartbeat(self, request, context) -> EmptyResponse:
        if self._validate_token(request.token, context):
            self.register_heartbeat(request.token)

            context.set_code(grpc.StatusCode.OK)
            return EmptyResponse()

    def Close(self, request, context) -> EmptyResponse:
        if self._validate_token(request.token, context):
            self.close_connection(request.token)

            context.set_code(grpc.StatusCode.OK)
            return EmptyResponse()

    def RequestModel(self, request, context) -> mila_pb2.Model:
        if self._validate_token(request.token, context):

            if self.should_wait_for_additional_clients():
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Waiting for more clients to join.")

            self.close_registration()
            if not self.are_more_rounds_required():
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "All rounds have been completed. Closing session.")

            if not self.set_client_status_to_awaiting_response(request.token):
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Next round is not available yet.")

            client = self._registry[request.token]
            logging.info("[{}] Sending Model (round={})".format(client, client.round))

            return mila_pb2.Model(
                json_configuration=self.get_configuration(),
                latest_checkpoint=self.get_latest_checkpoint()
            )

    def SendCheckpoint(self, request, context) -> EmptyResponse:
        if self._validate_token(request.token, context):
            with self.__lock:
                self.save_checkpoint(token=request.token, content=request.content)
                self.set_client_status_to_available(request.token)

                if self.are_all_updates_received():
                    self.aggregate()
                    self.enable_next_round()

            context.set_code(grpc.StatusCode.OK)
            return EmptyResponse()


class Server(IOManager):

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

        logging.info("Starting server at: [{}]".format(self._config.target))

        server.start()
        server.wait_for_termination()


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

        configuration_path = "{}/config.latest".format(self._config.save_path)
        with open(configuration_path, "w") as write_buffer:
            json.dump(configuration, write_buffer)

        return configuration_path

    def _retrieve_latest_file(self, folder_path: str) -> str:
        files = glob("{}/*.pt".format(folder_path))
        return max(files, key=os.path.getctime)

    def _train(self, configuration_path: str) -> str:
        config: Type[AbstractConfiguration] = self._reflect(self._config.config_type)
        config = config.from_json(configuration_path)

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

class Local(IOManager):
    def __init__(self, config: LocalConfiguration) -> None:
        self._config = config

    def get_checkpoints_paths(self):
        return [
            os.path.join(d, x)
            for d, dirs, files in os.walk(self._config.chekpoints_path)
            for x in files if x.endswith(".ckpt")
        ]

    def aggregate(self) -> None:
        logging.info("Start local aggregation")

        checkpoint_paths = self.get_checkpoints_paths()
        save_path = "{}/{}.aggregate.pt".format(self._config.save_path, self._config.current_round)

        aggregator: Type[AbstractAggregator] = self._reflect(self._config.aggregator_type)
        aggregator(**self._config.aggregator_options).run(checkpoint_paths=checkpoint_paths, save_path=save_path)

        logging.info("Aggregate model saved: [{}]".format(save_path))
    
    def run(self) -> None:
        # Box related stuffs maybe
        self.aggregate()


