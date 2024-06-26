from copy import deepcopy
import datetime
import logging
import os
import traceback
import sys
import time
from rich.logging import RichHandler


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
BOLD_SEQ = "\033[1m"

COLORS = {"WARNING": YELLOW, "INFO": WHITE, "DEBUG": BLUE, "CRITICAL": RED, "ERROR": RED}


class CustomFormatter(logging.Formatter):
    """
    Format time since first log to measure time of the overall run and add
    color to stdout
    """

    def __init__(self, *args, use_color=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.use_color = use_color

    def apply_colors(self, record):
        record = deepcopy(record)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + " " + levelname + " " + RESET_SEQ
            record.levelname = levelname_color
        return record

    def format(self, record):
        record = self.apply_colors(record)
        s = super().format(record)
        elapsed_seconds = record.created - self.start_time
        # using timedelta here for convenient default formatting
        elapsed = datetime.timedelta(seconds=elapsed_seconds)
        time_elapsed = str(elapsed).split(".")[0]
        return "{} | {}".format(time_elapsed, s)


class __Logger(logging.Logger):
    """
    Main logging class
    Contain one streamer (stdout)
    one file handler
    """

    _logger = logging.getLogger("logger_run")

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        self.now = datetime.datetime.now()
        formatter = self._get_formatter()
        if len(self._logger.handlers) != 0:
            self.stdout_handler = [h for h in self._logger.handlers if h.stream.name == "<stdout>"][0]
        else:
            self.stdout_handler = logging.StreamHandler(sys.stdout)
            self.stdout_handler.setFormatter(formatter)
            self._logger.addHandler(self.stdout_handler)
            self._logger.setLevel("DEBUG")

        self.propagate = False
        sys.excepthook = self.handle_excepthook

    def __getattr__(self, key):
        return getattr(self._logger, key)

    def handle_excepthook(self, type, message, stack):
        self._logger.error(
            f"An unhandled exception of type {type.__name__} occured: {message}. Traceback: \n{''.join(traceback.format_tb(stack))}"
        )

    def _get_formatter(self, **kwargs):
        return CustomFormatter(
            "%(asctime)s | [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s", datefmt="%Y-%m-%d %H:%M:%S", **kwargs
        )

    def add_file_log(self, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        path_log = os.path.join(dir_path, f"log_{self.now.strftime('%Y-%m-%d_%H-%M')}.log")
        self.file_handler = logging.FileHandler(path_log)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self._get_formatter(use_color=False))
        self._logger.addHandler(self.file_handler)

    def only_log_file(self, msg):
        self.disable_console_output()
        self._logger.log(self._logger.level, msg)
        self.enable_console_output()
        self._logger.setLevel(self._logger.level)

    def has_console_handler(self):
        return len([h for h in self.handlers if type(h) == logging.StreamHandler]) > 0

    def has_file_handler(self):
        return len([h for h in self.handlers if isinstance(h, logging.FileHandler)]) > 0

    def disable_console_output(self):
        if not self.has_console_handler():
            return
        self.removeHandler(self.stdout_handler)

    def enable_console_output(self):
        if self.has_console_handler():
            return
        self.addHandler(self.stdout_handler)

    def disable_file_output(self):
        if not self.has_file_handler():
            return
        self.removeHandler(self.file_handler)

    def enable_file_output(self):
        if self.has_file_handler():
            return
        self.addHandler(self.file_handler)

    def set_logging_level(self, level):
        self.level = level
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)


LOGGER = __Logger()
