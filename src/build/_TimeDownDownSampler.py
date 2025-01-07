from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
import pandas as pd
from typing import Union
from scipy import signal

LOG: logging.Logger = logging.getLogger(__name__)


class TimeDownSampler(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_channels: int = 1,
        sampling_factor: float = 0.1,
        copy: bool = True,
    ) -> None:
        # hyperparameter
        self.n_channels: int = n_channels
        self.sampling_factor: float = sampling_factor
        self.copy: bool = copy

        # init
        self.n_time_steps_: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        self.n_time_steps_ = X.shape[1] // self.n_channels

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        x = np.array(np.hsplit(X, self.n_channels))
        x = signal.resample(x, int(self.n_time_steps_ * self.sampling_factor), axis=2)
        x = np.hstack(x)

        LOG.debug("Resample from %s to %s", X.shape, x.shape)

        return x

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        x = np.array(np.hsplit(X, self.n_channels))
        x = signal.resample(x, self.n_time_steps_, axis=2)
        x = np.hstack(x)

        LOG.debug("Inverse Resample from %s to %s", X.shape, x.shape)

        return x
