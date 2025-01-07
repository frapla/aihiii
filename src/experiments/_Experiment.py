import logging
import sys
from pathlib import Path
from typing import List, Optional, Type
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.json_util as json_util
from src.utils.Csv import Csv
from src._Pipeline import Pipeline
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.experiments._Parameters import Parameters
from src.utils._ObjectChecker import ObjectChecker

LOG: logging.Logger = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self,
        user_pipeline: Type[BasePipe],
        processed_data_dir: Path,
        file_names_ai_in: List[str],
        file_names_ai_out: List[str],
        hyperparameter: Optional[dict] = None,
        used_columns_ai_in: Optional[List[str]] = None,
        used_columns_ai_out: Optional[List[str]] = None,
        feature_percentiles: Optional[List[int]] = None,
        target_percentiles: Optional[List[int]] = None,
        random_state_shuffle: Optional[int] = 42,
        shuffle_data: bool = True,
        used_ids_ai: Optional[List[int]] = None,
    ) -> None:
        """Utility to create and run an experiment

        Args:
            user_pipeline (Type[BasePipe]): BasePipe like user pipeline (NOT initialized)
            hyperparameter (dict): hyperparameter of user pipeline
            processed_data_dir (Path): directory of standardized data files
            file_names_ai_in (Optional[List[str]], optional): file names of data to feed into ai as input. Defaults to None.
            file_names_ai_out (Optional[List[str]], optional): file names of data to be predicted by ai. Defaults to None.
            feature_percentiles (Optional[List[int]], optional): dummy percentile(s) used for feature, None if features are DOE factors. Defaults to None.
            target_percentiles (Optional[List[int]], optional): dummy percentile(s) used for target, None if target are DOE factors. Defaults to None.
            random_state_shuffle (Optional[int], optional): random state of data shuffle in KFOLD, only active if shuffle_data=True. Defaults to 42.
            shuffle_data (bool, optional): shuffle data in KFOLD. Defaults to True.
            used_ids_ai (Optional[List[int]], optional): selection of simulation ids to use, use all if None. Defaults to None.
        """
        # environment
        self._hyperparameter: dict = hyperparameter
        self.str = StandardNames()
        self._used_ids_ai = used_ids_ai

        # init pipeline
        self.__user_pipeline: Type[BasePipe] = ObjectChecker().pipeline(pipe=user_pipeline)

        # files
        self._file_names_ai_in: List[str] = file_names_ai_in
        self._file_names_ai_out: List[str] = file_names_ai_out

        # feature target
        self._feature_percentiles: Optional[List[int]] = feature_percentiles
        self._target_percentiles: Optional[List[int]] = target_percentiles
        self._used_columns_ai_in: Optional[List[str]] = used_columns_ai_in
        self._used_columns_ai_out: Optional[List[str]] = used_columns_ai_out

        # shuffle KFOLD
        self._random_state_shuffle: Optional[int] = random_state_shuffle
        self._shuffle_data: bool = shuffle_data
        self._processed_data_dir: Path = processed_data_dir

    def prepare(self) -> None:
        """Prepare experiment"""
        # prepare directory
        Parameters().create(
            exp_dir=Path(),
            file_names_ai_in=self._file_names_ai_in,
            file_names_ai_out=self._file_names_ai_out,
            used_columns_ai_in=self._used_columns_ai_in,
            used_columns_ai_out=self._used_columns_ai_out,
            data_dir=self._processed_data_dir,
            pipeline_paras=self._hyperparameter,
            feature_percentiles=self._feature_percentiles,
            target_percentiles=self._target_percentiles,
            used_ids_ai=self._used_ids_ai,
        )

    def run(self) -> None:
        """Run main pipeline"""
        # init main pipeline
        LOG.info("Process Main Pipe")
        LOG.debug("Init Main Pipe")
        head_pipe = Pipeline(
            model_pipe=self.__user_pipeline,
            shuffle_data=self._shuffle_data,
            random_state_shuffle=self._random_state_shuffle,
        )

        # run
        LOG.info("Run Main Pipe")
        head_pipe.run()

    def get_score(self) -> float:
        cols = set(pd.read_csv(self.str.fname_results_csv).columns)
        metrics = pd.read_csv(self.str.fname_results_csv, index_col=[0, 1, 2] if self.str.target_class in cols else [0, 1])
        if self.str.target_class in cols:
            metric = metrics.loc[(slice(None), -1, "Test"), :].mean().mean()
        else:
            metric = metrics.loc[(-1, "Test"), :].mean()

        return metric
