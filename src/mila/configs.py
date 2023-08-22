import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import NamedTuple, List, Tuple, Any, Dict, Optional

@dataclass
class GrpcConfiguration:
    target: str = "localhost:8024"

    options: List[Tuple[str, Any]] = field(default_factory=lambda: [
        ("grpc.max_send_message_length", 1000000000),
        ("grpc.max_receive_message_length", 1000000000),
        ("grpc.ssl_target_name_override", "localhost")
    ])

    use_secure_connection: bool = False
    ssl_private_key: str = "data/certificates/client.key"
    ssl_cert: str = "data/certificates/client.crt"
    ssl_root_cert: str = "data/certificates/rootCA.pem"

@dataclass
class BoxConfiguration:
    box_configuration_path: str
    shared_dir_name: str
    save_path: str
    group_name: str


@dataclass
class ServerConfiguration:
    server_type: str
    server_manager_type: str
    task_configuration_file: str
    config_type: str
    executor_type: str

    aggregator_type: str = "mila.aggregators.PlainTorchAggregator"
    aggregator_options: Dict[str, Any] = field(default_factory=lambda: {})

    rounds_count: int = 10
    save_path: str = "data/logs/server/"
    start_point: Optional[str] = None

    workers: int = 2
    minimum_clients: int = 2
    maximum_clients: int = 100
    client_wait_time: int = 10
    heartbeat_timeout: int = 300

    blacklist: List[str] = field(default_factory=lambda: [])
    whitelist: List[str] = field(default_factory=lambda: [])
    use_whitelist: bool = False

    grpc_configuration: GrpcConfiguration = None
    box_configuration: BoxConfiguration = None

    @classmethod
    def from_json(cls, file_path: str) -> "ServerConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))
    
    def __post_init__(self):
        common_post_init(self)


@dataclass
class ClientConfiguration:
    name: str

    client_type: str
    executor_type: str = "kmol.run.Executor"

    config_type: str = "kmol.core.config.Config"
    
    save_path: str = "data/logs/client/"

    heartbeat_frequency: int = 60
    retry_timeout: int = 1
    model_overwrites: Dict[str, Any] = field(default_factory=lambda: {
        "output_path": "data/logs/local/",
        "epochs": 5
    })

    grpc_configuration: GrpcConfiguration = None
    box_configuration: BoxConfiguration = None


    @classmethod
    def from_json(cls, file_path: str) -> "ClientConfiguration":
        with open(file_path) as read_handle:
            return cls(**json.load(read_handle))
    
    def __post_init__(self):
        common_post_init(self)



def common_post_init(self):
    if self.box_configuration is not None:
            self.box_configuration = BoxConfiguration(**self.box_configuration)
    if self.grpc_configuration is not None:
        self.grpc_configuration = GrpcConfiguration(**self.grpc_configuration)
    
    save_path = Path(self.save_path)
    save_path = save_path / datetime.now().strftime('%Y-%m-%d_%H-%M')
    if not save_path.exists():
        save_path.mkdir(parents=True)
    self.save_path = save_path
    