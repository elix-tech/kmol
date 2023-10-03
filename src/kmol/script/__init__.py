from ..core.config import ScriptConfig
from ..core.helpers import SuperFactory
from mila.factories import AbstractScript
from mila.scripts import *
from .lr_finder import LrFinderScript
from .integrated_gradient import CaptumScript
from .protein_captum import ProteinCaptumScript
from .generate_msa import GenerateMsaScript


class ScriptLauncher:
    def __init__(self, config: ScriptConfig):
        self._config = config

    def run(self):
        SuperFactory.create(AbstractScript, self._config.script).run()
