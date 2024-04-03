from kmol.core.config import ScriptConfig
from kmol.core.helpers import SuperFactory
from mila.factories import AbstractScript
from mila.scripts import *
from .integrated_gradient import CaptumScript
from .protein_captum import ProteinCaptumScript, ProteinSequenceCaptumScript
from .generate_msa import GenerateMsaScript


class ScriptLauncher:
    def __init__(self, config: ScriptConfig):
        self._config = config

    def run(self):
        SuperFactory.create(AbstractScript, self._config.script).run()
