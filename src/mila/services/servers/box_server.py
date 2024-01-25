from threading import Thread
from kmol.core.logger import LOGGER as logging
from time import sleep
import traceback

from mila.configs import ServerConfiguration
from mila.services.server_manager.box_servicer import BoxServicer
from mila.services.servers.abstract_server import AbstractServer


class BoxServer(AbstractServer):
    def __init__(self, config: ServerConfiguration) -> None:
        self._config = config

    def _daemon(self, func, msg) -> None:
        logging.info(msg)
        while True:
            func()
            sleep(5)

    def _run(self, servicer: BoxServicer) -> None:
        servicer.send_configuration()

        authentication_worker = Thread(
            target=self._daemon, args=(servicer._authenticate, "Server authentification thread started")
        )
        authentication_worker.daemon = True
        authentication_worker.start()

        while not servicer._is_registration_closed:
            sleep(3)

        heartbeat_worker = Thread(target=self._daemon, args=(servicer._heartbeat, "Server heartbeat thread started"))
        heartbeat_worker.daemon = True
        heartbeat_worker.start()

        logging.info(f"Sending model for round {servicer._current_round}")
        servicer.send_model(mkdir=True)

        while servicer._current_round < self._config.rounds_count + 1:
            while not servicer._is_aggregate_ready():
                if servicer.get_clients_count() == 0:
                    logging.warning("There are no more client connected, ending the training")
                sleep(3)

            logging.info(f"Client Round {servicer._current_round} finished, downloading model")
            servicer.download_models()
            servicer.aggregate()
            logging.info(f"Sending aggregated model for round {servicer._current_round}")
            servicer.enable_next_round()
            servicer.send_model()

        logging.info(f"All round are finish")
        servicer.create_end_training_file()

    def run(self, servicer: BoxServicer):
        try:
            self._run(servicer)
        except KeyboardInterrupt:
            logging.info("Stopping gracefully...")
            servicer.create_end_training_file(msg="Training stop by user")
        except Exception as e:
            logging.info(f"An error occured, creating the end file")
            msg = str(e) + "\n" + "".join(traceback.format_exception(None, e, e.__traceback__))
            servicer.create_end_training_file(msg=msg)
            logging.error(e)
