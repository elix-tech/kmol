from argparse import ArgumentParser

from mila.configs import ServerConfiguration, ClientConfiguration
from mila.services.clients import AbstractClient
from mila.services.servers import AbstractServer
from mila.services.server_manager import ServerManager


class Executor:
    def __init__(self, config_path: str):
        self._config_path = config_path

    def run(self, job: str):
        if not hasattr(self, job):
            raise ValueError("Unknown job requested: {}".format(job))

        getattr(self, job)()

    def server(self) -> None:
        config: ServerConfiguration = ServerConfiguration.from_json(self._config_path)

        # server = Server(config=config)
        server = AbstractServer._reflect(config.server_type)(config=config)
        # servicer = DefaultServicer(config=config)
        servicer = ServerManager._reflect(config.server_manager_type)(config=config)
        server.run(servicer=servicer)

    def client(self) -> None:
        config: ClientConfiguration = ClientConfiguration.from_json(self._config_path)

        # client = Client(config=config)
        client = AbstractClient._reflect(config.client_type)(config=config)
        client.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    Executor(args.config).run(args.job)


if __name__ == "__main__":
    main()
