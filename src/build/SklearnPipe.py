from sklearn.utils import estimator_html_repr
from typing import Optional, Tuple, List
import pandas as pd
import logging
import sys
import pickle
import os
import json
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LinearRegression, LogisticRegression
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src.build.CheckPipe import CheckPipe
from src.utils.custom_log import init_logger
from src.build._TimeRobustScaler import TimeRobustScaler
from src.build._MultiChannelChain import MultichannelRegressorChain
from src.build._TimeDownDownSampler import TimeDownSampler

LOG: logging.Logger = logging.getLogger(__name__)


class SklearnPipe(BasePipe):
    def __init__(self) -> None:
        # init
        super().__init__()

        # hyperparameter
        self.core_estimator_name: str = "Dummy"
        self.core_estimator: Optional[BaseEstimator] = None
        self.core_estimator_hyperparameter: dict = {}
        self.tabular_feature_transformer: List[TransformerMixin] = []
        self.temporal_feature_transformer: List[TransformerMixin] = []
        self.target_transformer: List[TransformerMixin] = []
        self.target_temporal_compression_factor: float = 0.1
        self.feature_temporal_compression_factor: float = 0.1

        # misc
        self._is_multichannel_regression: bool = False
        self._time_stamps: Optional[pd.Index[float]] = None
        self._run_in_subprocess: bool = False

        # estimators
        self.estimators = {
            "Dummy": [DummyRegressor, DummyClassifier],
            "Linear": [LinearRegression, LogisticRegression],
            "DecisionTree": [DecisionTreeRegressor, DecisionTreeClassifier],
            "RandomForest": [RandomForestRegressor, RandomForestClassifier],
            "GradientBoosting": [GradientBoostingRegressor, GradientBoostingClassifier],
            "SVM": [SVR, SVC],
            "Ridge": [Ridge, RidgeClassifier],
            "XGBoost": [XGBRegressor, XGBClassifier],
        }

    def get_params(self) -> dict:
        """Read the parameters of the model

        Returns:
            dict: parameters
        """
        hyperparameter = self._model.get_params(deep=True)
        hyperparameter_out = {}
        for k, v in hyperparameter.items():
            try:
                json.dumps(v)
                hyperparameter_out[k] = v
            except TypeError:
                pass

        return hyperparameter_out

    def set_core_estimator(self, is_classification: bool = False) -> None:
        if is_classification:
            self.core_estimator = self.estimators[self.core_estimator_name][1]
        else:
            self.core_estimator = self.estimators[self.core_estimator_name][0]

        if self.core_estimator is not None:
            self.core_estimator = self.core_estimator(**self.core_estimator_hyperparameter)

    def fit(self, x: Data, y: Data) -> None:
        """Fit the model

        Args:
            x (Data): data container with input data
            y (Data): data container with target data
        """
        LOG.info("Train on temporal x %s and tabular x %s)", type(x.get_temporal()), type(x.get_tabular()))
        LOG.info("Train on temporal y %s and tabular y %s)", type(y.get_temporal()), type(y.get_tabular()))
        self._set_fitted(x=x, y=y)

        # prepare data
        x_tabular, x_temporal = self._prepare_features(x=x)

        # scaling
        tabular_feature_transform, temporal_feature_transform = self._feature_transformers(x=x)

        # input
        if tabular_feature_transform is not None and temporal_feature_transform is not None:
            input_pipe = make_column_transformer(
                (tabular_feature_transform, x_tabular.columns), (temporal_feature_transform, x_temporal.columns)
            )
            x_in = pd.concat([x_tabular, x_temporal], axis=1)
        elif tabular_feature_transform is not None:
            input_pipe = tabular_feature_transform
            x_in = x_tabular
        else:
            input_pipe = temporal_feature_transform
            x_in = x_temporal

        # target
        if isinstance(y.get_tabular(), pd.DataFrame):
            y_in = y.get_tabular()
            self._column_labels = y_in.columns
            y_vals = set(y_in.values.flatten())
            is_classification = y_vals == set([int(x) for x in y_vals])
            self.set_core_estimator(is_classification=is_classification)
            if len(self._column_labels) > 1:
                if is_classification:
                    LOG.info("Multilabel classification detected")
                    estimator = MultiOutputClassifier(estimator=self.core_estimator, n_jobs=os.cpu_count())
                    target_transformer = None
                else:
                    LOG.info("Multioutput regression detected")
                    target_transformer = make_pipeline(RobustScaler(), *self.target_transformer, memory=None)
                    estimator = TransformedTargetRegressor(
                        regressor=MultiOutputRegressor(estimator=self.core_estimator, n_jobs=os.cpu_count()),
                        transformer=target_transformer,
                    )
            else:
                if is_classification:
                    LOG.info("Single Label Classification Detected")
                    target_transformer = None
                    estimator = self.core_estimator
                else:
                    LOG.info("Single Output Regression Detected")
                    target_transformer = make_pipeline(RobustScaler(), *self.target_transformer, memory=None)
                    estimator = TransformedTargetRegressor(regressor=self.core_estimator, transformer=target_transformer)
        else:
            self.set_core_estimator(is_classification=False)
            self._is_multichannel_regression = True
            y_in = y.get_temporal().unstack(self._str.time)
            self._column_labels = y_in.columns
            self._time_stamps = y.get_temporal().index.get_level_values(self._str.time).drop_duplicates()
            n_tmps = y.get_temporal().index.get_level_values(self._str.time).nunique()

            if len(self._column_labels) > 1:
                LOG.info("Multichannel regression detected")
                chainer = MultichannelRegressorChain(base_estimator=self.core_estimator, n_time_steps=n_tmps)
            else:
                LOG.info("Channel regression detected")
                chainer = RegressorChain(base_estimator=self.core_estimator)

            target_transformer = make_pipeline(
                TimeRobustScaler(n_channels=y.get_temporal().shape[1], n_time_steps=n_tmps),
                TimeDownSampler(n_channels=y.get_temporal().shape[1], sampling_factor=self.target_temporal_compression_factor),
                *self.target_transformer,
                memory=None,
            )
            estimator = TransformedTargetRegressor(regressor=chainer, transformer=target_transformer, check_inverse=False)

        # fit
        self._model: Pipeline = make_pipeline(input_pipe, estimator, memory=None)
        LOG.info("Fit model with feature %s and target %s", x_in.shape, y_in.shape)
        self._model.fit(x_in, y_in)

    def predict(self, x: Data) -> Data:
        """Predict from the model

        Args:
            x (Data): data container with input data

        Returns:
            Data: data container with predicted data
        """
        self._check_fitted(x=x)
        LOG.info("Predict on temporal %s and tabular %s)", type(x.get_temporal()), type(x.get_tabular()))
        # treat input
        x_tabular, x_temporal = self._prepare_features(x=x)
        if x_tabular is not None and x_temporal is not None:
            x_in = pd.concat([x_tabular, x_temporal], axis=1)
        elif x_tabular is not None:
            x_in = x_tabular
        else:
            x_in = x_temporal

        # predict
        y_pred = self._model.predict(x_in)

        # transform back
        LOG.info("Transform back")
        y_out = Data()
        if isinstance(x.get_tabular(), pd.DataFrame):
            idx = x.get_tabular().index
        else:
            idx = x.get_temporal().index.get_level_values(self._str.id).drop_duplicates()
        if self._is_multichannel_regression:
            y_out.set_temporal(pd.DataFrame(y_pred, index=idx, columns=self._column_labels).stack())
        else:
            y_out.set_tabular(pd.DataFrame(y_pred, index=idx, columns=self._column_labels))

        return y_out

    def store(self) -> None:
        """Store the model"""
        LOG.info("Store model")
        with open("model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        with open("pipeline.html", "w") as f:
            f.write(estimator_html_repr(self._model))

        self._interface_names.store_params()

    def _prepare_features(self, x: Data) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if isinstance(x.get_tabular(), pd.DataFrame):
            x_tabular = x.get_tabular()
        else:
            x_tabular = None
        if isinstance(x.get_temporal(), pd.DataFrame):
            x_temporal = x.get_temporal().unstack(self._str.time)
            x_temporal.columns = [f"{c[0]}_{c[1]}" for c in x_temporal.columns]
        else:
            x_temporal = None

        return x_tabular, x_temporal

    def load(self, model_path: Path) -> None:
        """Load the model

        Args:
            model_path (Path): path to the model
        """
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        self._interface_names.load_params(inter_dir=model_path.parent)
        self._is_fitted = True

    def _feature_transformers(self, x: Data) -> Tuple[Optional[Pipeline], Optional[Pipeline]]:
        if isinstance(x.get_tabular(), pd.DataFrame):
            tabular_feature_transform = make_pipeline(RobustScaler(), *self.tabular_feature_transformer, memory=None)
        else:
            tabular_feature_transform = None
        if isinstance(x.get_temporal(), pd.DataFrame):
            temporal_feature_transform = make_pipeline(
                TimeRobustScaler(
                    n_channels=x.get_temporal().shape[1],
                    n_time_steps=x.get_temporal().index.get_level_values(self._str.time).nunique(),
                ),
                TimeDownSampler(
                    n_channels=x.get_temporal().shape[1],
                    sampling_factor=self.feature_temporal_compression_factor,
                ),
                *self.temporal_feature_transformer,
                memory=None,
            )
        else:
            temporal_feature_transform = None

        return tabular_feature_transform, temporal_feature_transform


if __name__ == "__main__":
    # init
    init_logger(log_lvl=logging.DEBUG)

    # run
    LOG.critical("Start Checks")
    checker = CheckPipe(pipe=SklearnPipe, n_samples=15)
    checker.run_checks_all(n_target_tsps_max=20)
    LOG.critical("Checks Done")
