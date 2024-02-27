import json
import os
import re
from glob import glob
from typing import Callable, Dict, Any

import optuna

from kmol.core.config import Config
from kmol.core.helpers import Loggable


class OptunaTemplateParser(Loggable):
    """
    Parser for dynamic configuration file templates.
    Will find placeholders wrapped in triple curly brackets "{{{...}}}" and generate suggestions for an Optuna trial:
    - each placeholder should have a name and some options, separated by an equal sign - ie:"{{{name=options}}}"
    - options separated by "|" will be categorical - ie: "{{{aggregate=mean|sum|max}}}"
    - numeric values should have 3 options separated by a dash "-". ie: {{{dropout=min|max|step}}}
        - The first value is the minimum value
        - The second value is the maximum value
        - The the third value is the incremental step between the minimum and the maximum
        - The [minimum, maximum] is a closed interval
    - if numeric values contain a dot ".", a float value will be suggested ie: {{{dropout=0.0-0.7-0.1}}}
    - if numeric values do not contain a dot ".", an int value will be suggested ie: {{{layers=2-5-1}}}
    """

    def __init__(self, config: Config, evaluator: Callable[[Config], float], log_path: str, delete_checkpoints: bool = True):
        Loggable.__init__(self, file_path=log_path)
        self.log("trial_number,performance,configuration_path\n")

        self._template = json.dumps(config.__dict__)
        self._template = self._template.replace(" ", "").replace("\n", "")

        self._evaluator = evaluator
        self._should_delete_checkpoints = delete_checkpoints

    def _get_trial_save_path(self, save_path: str, trial_id: int) -> str:
        return "{}/trial_{}/".format(save_path, trial_id)

    def _suggest_configuration(self, trial: optuna.Trial) -> Dict[str, Any]:
        replacements = {}

        for id_, key in enumerate(re.findall(r"{{{.*?}}}", self._template), start=1):
            name, placeholder = key[3:-3].split("=")

            if "|" in placeholder:
                replacements[key] = trial.suggest_categorical(name, placeholder.split("|"))
                continue

            key = '"{}"'.format(key)
            low, high, step = placeholder.split("-")

            if "." in placeholder:
                replacements[key] = trial.suggest_float(name=name, low=float(low), high=float(high), step=float(step))
            else:
                replacements[key] = trial.suggest_int(name=name, low=int(low), high=int(high), step=int(step))

        template = self._template
        for key, value in replacements.items():
            template = template.replace(key, str(value))

        return json.loads(template)

    def _store_trial_configuration(self, config: Dict[str, Any]) -> None:
        output_path = "{}.config.json".format(config["output_path"])
        with open(output_path, "w") as write_buffer:
            json.dump(config, write_buffer, indent=4)

    def _delete_checkpoints(self, save_path: str) -> None:
        for checkpoint_path in glob("{}checkpoint.*".format(save_path)):
            os.remove(checkpoint_path)

    def objective(self, trial: optuna.Trial) -> float:
        settings = self._suggest_configuration(trial)

        main_output_path = settings["output_path"]
        settings["output_path"] = self._get_trial_save_path(save_path=main_output_path, trial_id=trial.number)

        config = Config(**settings)
        self._store_trial_configuration(settings)

        result = self._evaluator(config)
        if self._should_delete_checkpoints:
            self._delete_checkpoints(settings["output_path"])

        self.log("{},{},{}.config.json\n".format(trial.number, result, settings["output_path"]))
        return result
