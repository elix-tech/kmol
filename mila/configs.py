import json
from typing import NamedTuple, List, Tuple, Any, Dict


class ServerConfiguration(NamedTuple):

    task_configuration_file: str
    aggregator_type: str
    config_type: str
    executor_type: str

    target: str = "127.0.0.1:8024"
    rounds_count: int = 10
    save_path: str = "data/logs/server/"

    workers: int = 2
    minimum_clients: int = 2
    maximum_clients: int = 100
    client_wait_time: int = 10
    heartbeat_timeout: int = 300

    use_secure_connection: bool = False
    ssl_private_key: str = "data/certificates/server.key"
    ssl_cert: str = "data/certificates/server.crt"
    ssl_root_cert: str = "data/certificates/rootCA.pem"
    options: List[Tuple[str, Any]] = [
        ("grpc.max_send_message_length", 1000000000),
        ("grpc.max_receive_message_length", 1000000000)
    ]

    blacklist: List[str] = []
    whitelist: List[str] = []
    use_whitelist: bool = False

    @classmethod
    def from_json(cls, file_path: str) -> "ServerConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))


class ClientConfiguration(NamedTuple):
    name: str
    config_type: str
    executor_type: str

    target: str = "127.0.0.1:8024"
    save_path: str = "data/logs/client/"

    heartbeat_frequency: int = 60
    model_overwrites: Dict[str, Any] = {
        "output_path": "data/logs/local/",
        "epochs": 5
    }

    use_secure_connection: bool = False
    ssl_private_key: str = "data/certificates/server.key"
    ssl_cert: str = "data/certificates/server.crt"
    ssl_root_cert: str = "data/certificates/rootCA.pem"

    @classmethod
    def from_json(cls, file_path: str) -> "ClientConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))
