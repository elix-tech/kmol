import os
import importlib
import uuid
from dataclasses import dataclass, field
from time import time
from typing import Callable, Dict, Type
import pytz
from datetime import datetime

from kmol.core.logger import LOGGER as logging
from ...exceptions import ClientAuthenticationError
from ...configs import ServerConfiguration
from ...factories import AbstractAggregator

class IOManager:

    def _read_file(self, file_path: str) -> bytes:
        with open(file_path, "rb") as read_buffer:
            return read_buffer.read()

    @classmethod
    def _reflect(cls, object_path: str) -> Callable:
        module, class_name = object_path.rsplit(".", 1)
        return getattr(importlib.import_module(module), class_name)


@dataclass
class Participant:
    name: str
    ip_address: str

    token: str = field(default_factory=lambda: str(uuid.uuid4()))
    round: int = 0
    awaiting_response: bool = False
    cut_from_training: bool = False

    __last_heartbeat: float = field(default_factory=datetime.now(pytz.utc).timestamp)

    def register_heartbeat(self, _time=None) -> None:
        if _time is None:
            _time = datetime.now(pytz.utc).timestamp()
        self.__last_heartbeat = _time

    def is_alive(self, heartbeat_timeout: float, _time=None) -> bool:
        if _time is None:
            _time = datetime.now(pytz.utc).timestamp()
        return (_time - self.__last_heartbeat) < heartbeat_timeout

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

    def register_heartbeat(self, token: str, _time=None) -> None:
        client = self._registry[token]

        client.register_heartbeat(_time)
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
        if not self._is_registration_closed:
            logging.info("Closing the registration")
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
            for client in self._registry.values() if client.is_alive(self._config.heartbeat_timeout)
        ]

    def get_client_filename_for_current_round(self, client):
        raise NotImplementedError
