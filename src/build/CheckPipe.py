from typing import Optional, Union, Type
import logging
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import string
import multiprocessing as mp

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src._StandardNames import StandardNames
from src.utils._ObjectChecker import ObjectChecker

LOG: logging.Logger = logging.getLogger(__name__)
RNG: np.random.Generator = np.random.default_rng(seed=42)
STR: StandardNames = StandardNames()


def run(pipe: Type[BasePipe], x: Data, y: Data, queue: mp.Queue, test_mode: bool = False) -> None:
    """Run pipeline in subprocess

    Args:
        pipe (Type[BasePipe]): not initialized pipeline
        x (Data): feature container
        y (Data): target container
        queue (mp.Queue): queue for communication
    """

    LOG.info("Init Pipeline")
    pipe = pipe(test_mode=test_mode)

    LOG.info("Fit Pipeline")
    pipe.fit(x=x, y=y)

    LOG.info("Predict Pipeline")
    y_pred: Data = pipe.predict(x=x)

    LOG.debug("Tabular Prediction:\n%s", y_pred.get_tabular())
    LOG.debug("Temporal Prediction:\n%s", y_pred.get_temporal())

    queue.put(y_pred)


class CheckPipe:
    def __init__(self, pipe: Type[BasePipe], n_samples: int = 10) -> None:
        """Standard check of pipeline with different data types randomly filled

        Args:
            pipe (Type[BasePipe]): not initialized pipeline
            n_samples (int, optional): sample size. Defaults to 10.
        """
        # pipeline
        self.pipe: Type[BasePipe] = ObjectChecker().pipeline(pipe=pipe)

        # data parameter
        self.n_samples: int = n_samples

        # container
        self._x: Data = Data()
        self._y: Data = Data()

    def run_checks_all(
        self,
        n_feature_channels_max: int = 2,
        n_feature_tsmps_max: int = 3,
        n_feature_labels_max: int = 4,
        n_target_classes_max: int = 3,
        n_target_labels_max: int = 3,
        n_target_tsps_max: int = 100,
        n_target_channels_max: int = 2,
    ) -> None:
        """Run checks for all feature and target types

        Args:
            n_feature_channels_max (int, optional): number of channels of temporal feature. Defaults to 2.
            n_feature_tsmps_max (int, optional): number of time stamps if temporal feature. Defaults to 3.
            n_feature_labels_max (int, optional): number of labels if tabular feature. Defaults to 4.
            n_target_classes_max (int, optional): number of classes if tabular targets. Defaults to 3.
            n_target_labels_max (int, optional): number of labels of tabular target. Defaults to 3.
            n_target_tsps_max (int, optional): number of time stamps of temporal target. Defaults to 100.
            n_target_channels_max (int, optional): number of channels of temporal target. Defaults to 2.
        """
        # check mixed input
        LOG.error("Check mixed input")
        self._x.reset_and_fill(
            tabular=self._get_tabular(n_columns=n_feature_labels_max),
            temporal=self._get_temporal(n_tsmps=n_feature_tsmps_max, n_channels=n_feature_channels_max),
        )
        self.run_checks_for_feature_type(
            n_classes_max=n_target_classes_max,
            n_labels_max=n_target_labels_max,
            n_tsps_max=n_target_tsps_max,
            n_channels_max=n_target_channels_max,
        )

        # check tabular input
        LOG.error("Check pure tabular input")
        self._x.reset_and_fill(
            tabular=self._get_tabular(n_columns=n_feature_labels_max),
            temporal=None,
        )
        self.run_checks_for_feature_type(
            n_classes_max=n_target_classes_max,
            n_labels_max=n_target_labels_max,
            n_tsps_max=n_target_tsps_max,
            n_channels_max=n_target_channels_max,
        )

        # check temporal input
        LOG.error("Check pure temporal input")
        self._x.reset_and_fill(
            tabular=None,
            temporal=self._get_temporal(n_tsmps=n_feature_tsmps_max, n_channels=n_feature_channels_max),
        )
        self.run_checks_for_feature_type(
            n_classes_max=n_target_classes_max,
            n_labels_max=n_target_labels_max,
            n_tsps_max=n_target_tsps_max,
            n_channels_max=n_target_channels_max,
        )

        LOG.error("Checks Done")

    def run_checks_for_feature_type(
        self, n_classes_max: int = 3, n_labels_max: int = 3, n_tsps_max: int = 100, n_channels_max: int = 2
    ) -> None:
        """Run checks for different target types

        Args:
            n_classes_max (int, optional): number of classes if tabular targets. Defaults to 3.
            n_labels_max (int, optional): number of labels of tabular target. Defaults to 3.
            n_tsps_max (int, optional): number of time stamps of temporal target. Defaults to 100.
            n_channels_max (int, optional): number of channels of temporal target. Defaults to 2.
        """
        LOG.warning("Check single label binary classification")
        self.check_tabular_prediction(n_labels=1, n_classes=1)

        LOG.warning("Check single label multiclass classification")
        self.check_tabular_prediction(n_labels=1, n_classes=n_classes_max)

        LOG.warning("Check multi label binary classification")
        self.check_tabular_prediction(n_labels=n_labels_max, n_classes=1)

        LOG.warning("Check multi label multiclass classification")
        self.check_tabular_prediction(n_labels=n_labels_max, n_classes=n_classes_max)

        LOG.warning("Check single label regression")
        self.check_tabular_prediction(n_labels=1, n_classes=np.inf)

        LOG.warning("Check multi label regression")
        self.check_tabular_prediction(n_labels=n_labels_max, n_classes=np.inf)

        LOG.warning("Check multichannel regression")
        self.check_temporal_prediction(n_tsps=n_tsps_max, n_channels=n_channels_max)

    def check_tabular_prediction(self, n_labels: int, n_classes: Union[int, float]) -> None:
        """Check prediction of tabular data

        Args:
            n_labels (int): number of labels (=number of table columns)
            n_classes (Union[int, float]): number of classes for classification or np.inf for regression
        """
        # data
        LOG.info("Set Data")
        self._y.reset_and_fill(
            tabular=self._get_tabular(n_columns=n_labels, n_levels=n_classes),
            temporal=None,
        )
        self._log_data()

        # pipe
        for test_mode in (True, False):
            LOG.info("Run Pipe - Test Mode: %s", test_mode)
            y_pred: Data = self._run_pipe(test_mode=test_mode)

            # checks
            self._log_check(
                same_columns=all(self._y.get_tabular().columns == y_pred.get_tabular().columns),
                same_index=all(self._y.get_tabular().index == y_pred.get_tabular().index),
            )

            if test_mode:
                if all(y_pred.get_tabular() == self._y.get_tabular()):
                    LOG.warning("Prediction is correct in test mode")
                else:
                    raise ValueError("Prediction is wrong in test mode")

    def check_temporal_prediction(self, n_tsps: int, n_channels: int) -> None:
        """Check prediction of temporal data

        Args:
            n_tsps (int): number of time steps
            n_channels (int): number of channels
        """
        # data
        LOG.info("Set Data")
        self._y.reset_and_fill(
            tabular=None,
            temporal=self._get_temporal(n_tsmps=n_tsps, n_channels=n_channels),
        )
        self._log_data()

        # pipe
        for test_mode in (True, False):
            LOG.info("Run Pipe - Test Mode: %s", test_mode)
            y_pred: Data = self._run_pipe(test_mode)
            # checks
            self._log_check(
                same_columns=all(self._y.get_temporal().columns == y_pred.get_temporal().columns),
                same_index=all(self._y.get_temporal().index == y_pred.get_temporal().index),
            )

            if test_mode:
                if all(y_pred.get_temporal() == self._y.get_temporal()):
                    LOG.warning("Prediction is correct in test mode")
                else:
                    raise ValueError("Prediction is wrong in test mode")

    def _log_data(self) -> None:
        """Log data information"""
        LOG.debug("Tabular Feature:\n%s", self._x.get_tabular())
        LOG.debug("Temporal Feature:\n%s", self._x.get_temporal())
        LOG.debug("Tabular Target:\n%s", self._y.get_tabular())
        LOG.debug("Temporal Target:\n%s", self._y.get_temporal())

    def _log_check(self, same_columns: bool, same_index: bool) -> None:
        """Log column and index equality

        Args:
            same_columns (bool):  indicator for column equality
            same_index (bool): indicator for index equality

        Raises:
            ValueError: Prediction has wrong columns
            ValueError: Prediction has wrong index
        """
        if same_columns:
            LOG.warning("Prediction has correct columns")
        else:
            raise ValueError("Prediction has wrong columns")
        if same_index:
            LOG.warning("Prediction has correct index")
        else:
            raise ValueError("Prediction has wrong index")

    def _run_pipe(self, test_mode: bool = False) -> Data:
        """Fit and predict data

        Returns:
            Data: data container with predicted data
        """
        queue = mp.Queue()
        proc = mp.Process(
            target=run, kwargs={"pipe": self.pipe, "x": self._x, "y": self._y, "queue": queue, "test_mode": test_mode}
        )
        proc.start()
        proc.join()
        y_pred: Data = queue.get()
        queue.close()
        proc.close()

        return y_pred

    def _get_temporal(self, n_tsmps: int, n_channels: int) -> pd.DataFrame:
        """Generate temporal data

        Args:
            n_tsmps (int): number of time steps
            n_channels (int): number of channels

        Returns:
            pd.DataFrame: random data, index with ID and TIME
        """
        data = RNG.random((self.n_samples * n_tsmps, n_channels))
        index = pd.MultiIndex.from_product([range(self.n_samples), range(n_tsmps)], names=[STR.id, STR.time])
        columns = list(string.ascii_uppercase[:n_channels])

        return pd.DataFrame(data=data, index=index, columns=columns)

    def _get_tabular(self, n_columns: int, n_levels: Union[int, float] = np.inf) -> pd.DataFrame:
        """Generate tabular data

        Args:
            n_columns (int): number of columns
            n_levels (Union[int, float], optional): number of classes of continuous. Defaults to np.inf.

        Returns:
            pd.DataFrame: table with random data, index is ID
        """
        if n_levels == np.inf:
            data = RNG.random((self.n_samples, n_columns))
        else:
            data = RNG.integers(0, n_levels, (self.n_samples, n_columns), endpoint=True)
        index = pd.Index(range(self.n_samples), name=STR.id)
        columns = list(string.ascii_uppercase[:n_columns])

        return pd.DataFrame(data=data, index=index, columns=columns)
