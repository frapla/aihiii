import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Type, Union

import optuna

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.experiments._Experiment import Experiment
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator
from src.tuner.Hyperparameter import Hyperparameter
from src.utils._ObjectChecker import ObjectChecker

LOG: logging.Logger = logging.getLogger(__name__)


class EndWhenPauseFile:
    def __init__(self, study_dpath: Path):
        self.study_dpath: Path = study_dpath
        self.stop_fpath: Path = study_dpath / "stop.txt"

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.stop_fpath.is_file():
            LOG.warning("Stop file found, stopping study")
            study.stop()
            LOG.debug("Remove stop file %s", self.stop_fpath)
            self.stop_fpath.unlink()


class Study:
    def __init__(
        self,
        user_pipeline: Type[BasePipe],
        hyperparameter_generator: BaseHyperparameterGenerator,
        n_trials: int = 2,
        study_name: str = "Study",
        sampler: Union[optuna.samplers.BaseSampler, None] = None,
        pruner: Union[optuna.pruners.BasePruner, None] = None,
        timeout: Union[float, None] = None,
        n_jobs: int = 1,
    ) -> None:
        """Prepare and run an optuna study

        Args:
            user_pipeline (BasePipe): user pipeline object (NOT initialized)
            hyperparameter_generator (BaseHyperparameterGenerator): generator of hyperparameter dict for each trial
            n_trials (int, optional): number of trials. Defaults to 2.
            study_name (str, optional): name of study. Defaults to "Study".
            sampler (Union[optuna.samplers.BaseSampler, None], optional): optuna sampler. Defaults to None.
            pruner (Union[optuna.pruners.BasePruner, None], optional): optuna pruner. Defaults to None.
            timeout (Union[float, None], optional): timeout for optuna optimizer. Defaults to None.
            n_jobs (int, optional): number of parallel processes. Defaults to 1.
        """
        # user pipeline
        self.__pipeline = user_pipeline
        self.__generator = ObjectChecker().hyperparameter_generator(generator=hyperparameter_generator)

        # optuna
        self.__n_trials = n_trials
        self.__study_name = study_name
        self.__sampler = sampler
        self.__pruner = pruner
        self.__timeout = timeout
        self.__n_jobs = n_jobs

    def run_trial(self, trial: optuna.Trial) -> Union[float, Tuple[float, float, float, float]]:
        """Run trial

        Args:
            trial (optuna.Trial): trial object

        Returns:
            float: objective value for optimization
        """
        # set experiments directory name
        cwd = os.getcwd()
        trial_dir = Path(f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_trial_{trial.number}")
        trial_dir.mkdir(exist_ok=True)
        os.chdir(trial_dir)

        params: Hyperparameter = self.__generator.suggest_hyperparameter(trial=trial)

        # init experiment
        experiment_handler = Experiment(
            user_pipeline=self.__pipeline,
            processed_data_dir=Path("..") / ".." / ".." / "data" / "doe" / params.database,
            file_names_ai_in=params.file_names_ai_in,
            file_names_ai_out=params.file_names_ai_out,
            feature_percentiles=params.feature_percentiles,
            target_percentiles=params.target_percentiles,
            used_columns_ai_out=params.used_columns_ai_out,
            used_columns_ai_in=params.used_columns_ai_in,
            hyperparameter=params.estimator_hyperparameter,
            shuffle_data=params.shuffle_data,
            random_state_shuffle=params.random_state_shuffle,
            used_ids_ai=params.used_ids_ai,
        )

        # create experiment
        experiment_handler.prepare()

        # run
        experiment_handler.run()

        # eval
        metric_test = experiment_handler.get_score()

        os.chdir(cwd)
        return metric_test

    def run_study(self) -> optuna.Study:
        """Run optuna study

        Returns:
            optuna.Study: filled study object
        """
        # set URL of data base
        db_url = f"sqlite:///{Path().absolute() / 'study.sqlite3'}"
        LOG.info("URL of Optuna Database is %s", db_url)
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        # set direction
        direction = optuna.study.StudyDirection.MAXIMIZE

        # init study
        study = optuna.create_study(
            study_name=self.__study_name,
            direction=direction,
            storage=db_url,
            load_if_exists=True,
            sampler=self.__sampler,
            pruner=self.__pruner,
        )

        # run
        stopper = EndWhenPauseFile(study_dpath=Path().absolute())
        study.optimize(func=self.run_trial, n_trials=self.__n_trials, timeout=self.__timeout, n_jobs=self.__n_jobs, callbacks=[stopper])

        return study
