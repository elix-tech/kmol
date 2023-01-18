from time import time
from pathlib import Path

from kmol.core.logger import LOGGER as logging

from .server_manager import ServerManager, Participant
from ..box_utils import Box, File, User
from ...configs import ServerConfiguration


class BoxServicer(ServerManager):
    def __init__(self, config: ServerConfiguration) -> None:
        super().__init__(config)
        self.box = Box(self._config.box_configuration, role="server")

    def register_client(self, name: str, user: User) -> str:
        client = Participant(name=name, ip_address="", token=user.id)
        client.register_heartbeat(self.box.get_last_hb_user(user))
        for entry in self._registry.values():
            if client == entry:
                return entry.token

        self._registry[client.token] = client
        self._last_registration_time = time()

        logging.info("[{}] Successfully authenticated (clients={})".format(client, self.get_clients_count()))
        return client.token
    
    def _authenticate(self):
        auth_files = [f.get() for f in self.box.auth_dir.get_items()]
        for file in auth_files:
            if file.content().decode() == "":
                if self._is_registration_closed:
                    self.box.update_text(file, f"Error - Authentication failed... Registration is closed.")
                else:
                    user = file.created_by
                    token = self.register_client(user.name, user)
                    self.box.update_text(file, f"Authenticated - Token {token}")
        if not self.should_wait_for_additional_clients():
            self.close_registration()


    def _heartbeat(self):
        for file in self.box.hb_dir.get_items():
            file = file.get()
            user = file.created_by
            time_last_hb = self.box.get_modify_time_file(file)
            if self._validate_token(user.id, file):
                self.register_heartbeat(user.id, time_last_hb)
            else:
                if not self._registry[user.id].cut_from_training:
                    logging.warning(f"{str(self._registry[user.id])}: Lost connection continuing training without this client")
                    folder = self.box.get_folder_from_path(self.box.base_path)
                    name = self._registry[user.id].name
                    self.box.upload_text(folder, f"{name}.lost_connection", f"lost connection on round {self._current_round}")
                    self._registry[user.id].cut_from_training = True


    def _validate_token(self, token: str, file: File) -> bool:
        return self._registry[token].is_alive(self._config.heartbeat_timeout)

    def send_configuration(self):
        folder = self.box.get_folder_from_path(self.box.base_path, mkdir=True)
        self.box.upload_file("cfg.json", self._config.task_configuration_file, folder)

    def enable_next_round(self) -> None:
        super().enable_next_round()
        self.box.get_folder_from_path(self.box.base_path / "rounds" / f"round_{self._current_round}", mkdir=True)

    def send_model(self, mkdir=False):
        folder = self.box.get_folder_from_path(self.box.base_path / "rounds" / f"round_{self._current_round}", mkdir)
        if self._latest_checkpoint is not None:
            self.box.upload_file("server.agg", self._latest_checkpoint, folder)
        else:
            # Should only happen if start_point is not provided
            self.box.upload_text(folder, "no_starting_model.info", "")

    def _is_aggregate_ready(self,):  
        nb_checkpont = self.box.count_checkpoints(self.box.base_path / "rounds" / f"round_{self._current_round}")
        return nb_checkpont == self.get_clients_count()
    
    def get_clients_count(self) -> int:
        return sum(1 for client in self._registry.values() if not client.cut_from_training)

    def download_models(self):
        folder = self.box.get_folder_from_path(self.box.base_path / "rounds" / f"round_{self._current_round}")
        self.box.download_all_checkpoints(Path(self._config.save_path) / f"round_{self._current_round}", folder)

    def get_client_filename_for_current_round(self, client: Participant):
        return Path(self._config.save_path) / f"round_{self._current_round}" / f"{str(client.name).lower()}.pt"
    

    def create_end_training_file(self, msg=f"Training finished {str(time())}"):
        file_name = "server_ended.txt"
        folder = self.box.get_folder_from_path(self.box.base_path)
        hb_file = self.box.upload_text(folder, file_name, msg)