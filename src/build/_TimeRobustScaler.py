from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
import pandas as pd
from typing import Tuple, Union

LOG: logging.Logger = logging.getLogger(__name__)


class TimeRobustScaler(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_time_steps: int = 1,
        n_channels: int = 1,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
        copy: bool = True,
    ) -> None:
        # hyperparameter
        self.n_time_steps: int = n_time_steps
        self.n_channels: int = n_channels
        self.quantile_range: Tuple[float, float] = quantile_range
        self.with_centering: bool = with_centering
        self.with_scaling: bool = with_scaling
        self.copy: bool = copy

        # init
        self.n_features_in_: int = n_time_steps * n_channels
        self.center_: np.ndarray = np.zeros(self.n_features_in_)
        self.scale_: np.ndarray = np.zeros(self.n_features_in_)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        if isinstance(X, pd.DataFrame):
            x = X.values
        else:
            x = X

        # center
        if self.with_centering:
            center = [np.median(x[:, i : i + self.n_time_steps]) for i in range(0, X.shape[1], self.n_time_steps)]
            self.center_ = np.array([[p] * self.n_time_steps for p in center]).flatten()

        # iqr
        if self.with_scaling:
            iqr = [
                np.percentile(x[:, i : i + self.n_time_steps], self.quantile_range[1])
                - np.percentile(x[:, i : i + self.n_time_steps], self.quantile_range[0])
                for i in range(0, x.shape[1], self.n_time_steps)
            ]
            self.scale_ = np.array([[p] * self.n_time_steps for p in iqr]).flatten()

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        x = X.copy() if self.copy else X

        if self.with_centering:
            x -= self.center_
        if self.with_scaling:
            x /= self.scale_
        return x

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        x = X.copy() if self.copy else X

        if self.with_scaling:
            x *= self.scale_
        if self.with_centering:
            x += self.center_
        return x
