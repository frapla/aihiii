from typing import Optional
import pandas as pd
import logging
import sys
from pathlib import Path
from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src.build.CheckPipe import CheckPipe
from src.utils.custom_log import init_logger

LOG: logging.Logger = logging.getLogger(__name__)


class BaseLineUniversal(BasePipe):
    def __init__(self) -> None:
        # init
        super().__init__()

    def fit(self, x: Data, y: Data) -> None:
        """Fit the model

        Args:
            x (Data): data container with input data
            y (Data): data container with target data
        """
        self._set_fitted(x=x, y=y)
        LOG.debug("Train on temporal x %s and tabular x %s)", type(x.get_temporal()), type(x.get_tabular()))
        LOG.debug("Train on temporal y %s and tabular y %s)", type(y.get_temporal()), type(y.get_tabular()))
        self._interface_names.set_features(x)
        self._interface_names.set_target(y)

        # determine task type
        y: pd.DataFrame = self.__set_model(y_temporal=y.get_temporal(), y_tabular=y.get_tabular())

        # treat input
        x: pd.DataFrame = self.__treat_input(x)

        # ensure order
        ref_ids = y.index.get_level_values(self._str.id).to_list()
        self._rng.shuffle(ref_ids)
        x = x.loc[ref_ids]
        y = y.loc[ref_ids]

        # fit
        self._model.fit(x, y)
        self._column_labels = y.columns
        self._is_fitted = True


    def predict(self, x: Data) -> Data:
        """Predict from the model

        Args:
            x (Data): data container with input data

        Returns:
            Data: data container with predicted data
        """
        self._check_fitted(x=x)

        LOG.debug("Predict on temporal %s and tabular %s)", type(x.get_temporal()), type(x.get_tabular()))
        pred = Data()
        # treat input
        x: pd.DataFrame = self.__treat_input(x)

        # predict
        if self._is_fitted:
            y_pred = self._model.predict(x)
            y_pred = pd.DataFrame(y_pred, columns=self._column_labels, index=x.index)
            if isinstance(self._column_labels, pd.MultiIndex):
                LOG.debug("Multichannel regression detected - return temporal data")
                pred.set_temporal(y_pred.stack(self._str.time, future_stack=True))
            else:
                LOG.debug("Regression / Classification detected - return tabular data")
                pred.set_tabular(y_pred)
        else:
            LOG.critical("Model is not fitted - return None")

        LOG.debug("Predicted temporal %s, tabular%s)", type(pred.get_temporal()), type(pred.get_tabular()))

        return pred

    def __treat_input(self, x: Data) -> Optional[pd.DataFrame]:
        if isinstance(x.get_tabular(), pd.DataFrame):
            return x.get_tabular()
        else:
            LOG.warning("Temporal input not supported -generate dummy from temporal")
            xx = x.get_temporal().unstack(self._str.time)
            return pd.DataFrame(np.zeros((xx.shape[0], 1)), index=xx.index)

    def __set_model(self, y_temporal: Optional[pd.DataFrame], y_tabular: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if y_temporal is None and isinstance(y_tabular, pd.DataFrame):
            if set(y_tabular.values.flatten()) == set([int(x) for x in y_tabular.values.flatten()]):
                LOG.debug("Classification detected")
                self._model: DummyClassifier = DummyClassifier(strategy="stratified", random_state=42)
            else:
                LOG.debug("Regression detected")
                self._model: DummyRegressor = DummyRegressor(strategy="median")
            y = y_tabular
        elif isinstance(y_temporal, pd.DataFrame) and y_tabular is None:
            LOG.debug("Multichannel regression detected")
            y = y_temporal.unstack(self._str.time)
            self._model: DummyRegressor = DummyRegressor(strategy="median")
        else:
            LOG.error("Mix of temporal and tabular data not supported - Set Target to None")
            y = None

        return y


if __name__ == "__main__":
    # init
    init_logger(log_lvl=logging.WARNING)

    # run
    LOG.critical("Start Checks")
    checker = CheckPipe(pipe=BaseLineUniversal)
    checker.run_checks_all()
    LOG.critical("Checks Done")
