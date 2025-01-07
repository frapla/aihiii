import sys
from pathlib import Path
from typing import Union, Any, Optional, List, Tuple
import logging
import pandas as pd
import numpy as np
import pickle

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src._StandardNames import StandardNames
from src.evaluate._Data import Data
from src.build._InterfaceNames import InterfaceNames

LOG: logging.Logger = logging.getLogger(__name__)


class BasePipe:
    def __init__(self) -> None:
        """Parent class for data transformation and machine learning wrapper

        Args:
            data_identifier (str): identifier for data set (e.g. number of fold) default is "0"
        """
        self._is_fitted: bool = False
        self._str: StandardNames = StandardNames()
        self._model: Optional[Any] = None
        self._column_labels: Optional[Union[pd.Index, pd.MultiIndex]] = None
        self._rng = np.random.default_rng(seed=42)
        self._run_in_subprocess: bool = True
        self._interface_names: InterfaceNames = InterfaceNames()

    def set_params(self, **parameters):
        """Set hyperparameters

        Returns:
            _type_: _description_
        """
        for parameter, value in parameters.items():
            if parameter in self.__dict__ and self.__valid_para_name(parameter):
                LOG.debug("Set parameter %s to value %s", parameter, value)
                setattr(self, parameter, value)
            else:
                LOG.warning("Parameter %s not in namespace - IGNORE", parameter)
        return self

    def get_params(self) -> dict:
        """Get hyperparameters

        Returns:
            dict: hyperparameters
        """
        return {key: value for key, value in self.__dict__.items() if self.__valid_para_name(key)}

    def __valid_para_name(self, para: str) -> bool:
        """Check naming convention for names of parameters

        Args:
            para (str): name of parameter object

        Returns:
            bool: check result
        """
        return not para.startswith("_")

    def fit(self, x: Data, y: Data) -> None:
        """Fit the model

        Args:
            x (Data): data container with input data
            y (Data): data container with target data
        """
        self._interface_names.set_features(x)
        self._interface_names.set_target(y)
        self._is_fitted = True

    def predict(self, x: Data) -> Data:
        """Predict from the model

        Args:
            x (Data): data container with input data

        Returns:
            Data: data container with predicted data
        """
        if self._is_fitted and self._interface_names.compare_features(x):
            return Data()
        else:
            LOG.critical("Model not fitted or feature names does not match - EXIT")
            sys.exit()

    def store(self) -> None:
        with open("model.pkl", "wb") as file:
            pickle.dump(self._model, file)

        self._interface_names.store_params()

    def load(self) -> None:
        with open("model.pkl", "rb") as file:
            self._model = pickle.load(file)

        self._interface_names.load_params()
        self._is_fitted = True

    def _check_fitted(self, x: Data) -> None:
        if not self._is_fitted:
            LOG.critical("Model not fitted - EXIT")
            sys.exit()
        if not self._interface_names.compare_features(x):
            LOG.critical("Feature names does not match - EXIT")
            sys.exit()

    def _set_fitted(self, x: Data, y: Data) -> None:
        self._is_fitted = True
        self._interface_names.set_features(x)
        self._interface_names.set_target(y)
