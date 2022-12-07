from time import time

from kmol.core.logger import LOGGER as logging

from .server_manager import ServerManager, Participant
from ...exceptions import ClientAuthenticationError
from ..clients.box_utils import Box, File, User
from ...configs import ServerConfiguration


class BoxServicer(ServerManager):
    def __init__(self, config: ServerConfiguration) -> None:
        super().__init__(config)
        self.box = Box(self._config.box_configuration)

    def register_client(self, name: str, user: User) -> str:

        client = Participant(name=name, ip_address="", token=user.id)
        client.register_heartbeat(self.box.get_last_hb_user(user))
        for entry in self._registry.values():
            if client == entry:
                return entry.token

        if not self.should_wait_for_additional_clients():
            raise ClientAuthenticationError("Authentication failed... Registration is closed.")

        self._registry[client.token] = client
        self._last_registration_time = time()

        logging.info("[{}] Successfully authenticated (clients={})".format(client, self.get_clients_count()))
        return client.token
    
    def _authenticate(self):
        auth_files = [f for f in self.box.auth_dir.get_items()]
        for file in auth_files:
            if file.content().decode() == "":
                user = file.created_by
                token = self.register_client(user.name, user)
                self.box.update_text(file, f"Authenticated - Token {token}")

    def _heartbeat(self):
        for file in self.box.hb_dir.get_items():
            user = file.created_by
            time_last_hb = self.box.get_modify_time_file(file)
            if self._validate_token(user.id, file, time_last_hb):
                self.register_heartbeat(user.id, time, time_last_hb)


    def _validate_token(self, token: str, file: File, _time) -> bool:
        return self._registry[token].is_alive(self._config.heartbeat_timeout, _time)

    
    def _request_model(self, request, context):
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