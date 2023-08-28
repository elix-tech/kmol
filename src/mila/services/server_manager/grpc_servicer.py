from threading import Lock

import grpc
from google.protobuf.empty_pb2 import Empty as EmptyResponse
from kmol.core.logger import LOGGER as logging

from ...configs import ServerConfiguration
from ...exceptions import ClientAuthenticationError
from ...protocol_buffers import mila_pb2, mila_pb2_grpc
from .server_manager import ServerManager, Participant


class GrpcServicer(ServerManager, mila_pb2_grpc.MilaServicer):

    def __init__(self, config: ServerConfiguration) -> None:
        super().__init__(config=config)
        self.__lock = Lock()

    def _validate_token(self, token: str, context) -> bool:
        if not self.verify_token(token=token, ip_address=self._get_ip(context)):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Access Denied... Token is invalid...")

        return True

    def _get_ip(self, context) -> str:
        return context.peer().split(":")[1]

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

            return mila_pb2.Model(json_configuration=self.get_configuration(), latest_checkpoint=self.get_latest_checkpoint())

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


    def get_client_filename_for_current_round(self, client: Participant):
        return "{}/{}.{}.{}.remote".format(
            self._config.save_path, client.name, client.ip_address.replace(".", "_"), self._current_round
        )