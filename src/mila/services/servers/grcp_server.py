from concurrent import futures

import grpc
from kmol.core.logger import LOGGER as logging

from ...configs import ServerConfiguration
from ...protocol_buffers import mila_pb2_grpc
from .abstract_server import AbstractServer

class Server(AbstractServer):

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

