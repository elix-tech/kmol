import logging
from argparse import ArgumentParser

from .configs import ServerConfiguration, ClientConfiguration
from .services import Server, Client, DefaultServicer

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class Executor:

    def __init__(self, config_path: str):
        self._config_path = config_path

    def run(self, job: str):
        if not hasattr(self, job):
            raise ValueError("Unknown job requested: {}".format(job))

        getattr(self, job)()

    def server(self) -> None:
        config: ServerConfiguration = ServerConfiguration.from_json(self._config_path)

        server = Server(config=config)
        servicer = DefaultServicer(config=config)

        server.run(servicer=servicer)

    def client(self) -> None:
        config: ClientConfiguration = ClientConfiguration.from_json(self._config_path)

        client = Client(config=config)
        client.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("config")
    args = parser.parse_args()

    Executor(args.config).run(args.job)


if __name__ == "__main__":
    main()
