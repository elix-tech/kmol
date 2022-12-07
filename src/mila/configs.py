import json
from typing import NamedTuple, List, Tuple, Any, Dict, Optional


class GrcpConfiguration(NamedTuple):
    target: str = "localhost:8024"

    options: List[Tuple[str, Any]] = [
        ("grpc.max_send_message_length", 1000000000),
        ("grpc.max_receive_message_length", 1000000000),
        ("grpc.ssl_target_name_override", "localhost")
    ]

    use_secure_connection: bool = False
    ssl_private_key: str = "data/certificates/client.key"
    ssl_cert: str = "data/certificates/client.crt"
    ssl_root_cert: str = "data/certificates/rootCA.pem"

class BoxConfiguration(NamedTuple):
    box_configuration_path: str
    shared_dir_name: str
    save_path: str
    group_name: str
    scan_frequency: int

class ServerConfiguration(NamedTuple):

    task_configuration_file: str
    config_type: str
    executor_type: str

    aggregator_type: str
    aggregator_options: Dict[str, Any] = {}

    rounds_count: int = 10
    save_path: str = "data/logs/server/"
    start_point: Optional[str] = None

    workers: int = 2
    minimum_clients: int = 2
    maximum_clients: int = 100
    client_wait_time: int = 10
    heartbeat_timeout: int = 300

    blacklist: List[str] = []
    whitelist: List[str] = []
    use_whitelist: bool = False

    grcp_configuration: GrcpConfiguration = None
    bbox_configuration: BoxConfiguration = None

    @classmethod
    def from_json(cls, file_path: str) -> "ServerConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))


class ClientConfiguration(NamedTuple):
    name: str
    executor_type: str = "kmol.run.Executor"

    config_type: str = "kmol.core.config.Config"
    
    save_path: str = "data/logs/client/"

    heartbeat_frequency: int = 60
    retry_timeout: int = 1
    model_overwrites: Dict[str, Any] = {
        "output_path": "data/logs/local/",
        "epochs": 5
    }

    grcp_configuration: GrcpConfiguration = None
    bbox_configuration: BoxConfiguration = None


    @classmethod
    def from_json(cls, file_path: str) -> "ClientConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))



