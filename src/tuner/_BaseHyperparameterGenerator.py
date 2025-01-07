import logging

import optuna

LOG: logging.Logger = logging.getLogger(__name__)


class BaseHyperparameterGenerator:
    def __init__(self) -> None:
        """Base class for optuna hyperparameter generator"""

    def suggest_hyperparameter(self, trial: optuna.Trial) -> dict:
        """Draw new hyperparameter for each trial, suggestions have to be compatible with chosen sampler

        Args:
            trial (optuna.Trial): current trial

        Returns:
            dict: hyperparameter
        """
        params = {"example": trial.suggest_categorical("example", [None])}

        return params
