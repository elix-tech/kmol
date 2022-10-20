from ..core.config import ScriptConfig
from ..core.helpers import SuperFactory
from mila.factories import AbstractScript
from mila.scripts import *

class ScriptLauncher:
    def __init__(self, config: ScriptConfig):
        self._config = config

    def run(self):
        SuperFactory.create(AbstractScript, self._config.script).run()