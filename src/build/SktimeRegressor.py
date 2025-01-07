from typing import Optional, List, Union, Set
import logging
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import numpy as np
import sys
from pathlib import Path
import pickle
import pandas as pd
from sktime.regression.base import BaseRegressor
import inspect


src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src.build.CheckPipe import CheckPipe
from src.utils.custom_log import init_logger
import src.utils.json_util as json_util
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class SktimeRegressor(BasePipe):
    def __init__(self, test_mode: bool = False) -> None:
        """Universal ANN with Keras"""
        # init
        super().__init__()

        # regressors
        self._regressors: Set[str] = {
            "CNNRegressor",
            "CNTCRegressor",
            "FCNRegressor",
            "LSTMFCNRegressor",
            "MLPRegressor",
            "ResNetRegressor",
            "InceptionTimeRegressor",
            "MACNNRegressor",
            "MCDCNNRegressor",
            "SimpleRNNRegressor",
            "TapNetRegressor",
            "KNeighborsTimeSeriesRegressor",
            "TimeSeriesForestRegressor",
            "TimeSeriesSVRTslearn",
            "RocketRegressor",
        }

        # parameter
        self.regressor_name: str = "CNNRegressor"
        self.hyperparameters: dict = {}
        self.temporal_feature_n_tsps: int = 140

        # misc
        self._loss: Optional[Union[str, List[str]]] = None
        self.__test: bool = test_mode
        self.__y_store: Optional[np.ndarray] = None

        # out types
        self._n_classes: int = 0
        self._is_regression: bool = False
        self._is_multi_channel_regression: bool = False
        self._label_names: Optional[pd.Index[str]] = None
        self._time_stamps: Optional[pd.Index[float]] = None

        # model
        self._transformer_target: Optional[Union[OneHotEncoder, RobustScaler]] = None
        self._transformer_feature_tabular: Optional[RobustScaler] = None
        self._transformer_feature_temporal: Optional[RobustScaler] = None

    def get_regressor_names(self) -> List[str]:
        """Get the available regressors

        Returns:
            List[str]: regressor names
        """
        return sorted(self._regressors)

    def get_params(self) -> dict:
        """Read the parameters of the model

        Returns:
            dict: parameters
        """
        h_para = {
            "set_params": {p: v for p, v in self.__dict__.items() if not p.startswith("_")},
        }

        h_para["transformer_target"] = None if self._transformer_target is None else self._transformer_target.get_params()
        h_para["transformer_feature_tabular"] = (
            None if self._transformer_feature_tabular is None else self._transformer_feature_tabular.get_params()
        )
        h_para["transformer_feature_temporal"] = (
            None if self._transformer_feature_temporal is None else self._transformer_feature_temporal.get_params()
        )

        return h_para

    def fit(self, x: Data, y: Data) -> None:
        """Fit the model

        Args:
            x (Data): data container with input data
            y (Data): data container with target data
        """
        # init
        self._set_fitted(x=x, y=y)

        # set data
        LOG.debug("Set data for fitting")
        x_in = self._prepare_input(x=x, training=True)
        y_in = self._prepare_output(y=y)

        for i, x_in_i in enumerate(x_in):
            LOG.debug("Pre-Processed Input %s %s:\n%s", i, x_in_i.shape, x_in_i)
        for i, y_in_i in enumerate(y_in):
            LOG.debug("Pre-Processed Target %s %s:\n%s", i, y_in_i.shape, y_in_i)

        # set model
        LOG.debug("Build model")
        self._model: BaseRegressor = self._set_estimator()

        # training
        LOG.debug("Start training")
        self._model.fit(x_in, y_in)
        if self.__test:
            self.__y_store = y_in

        LOG.debug("Training done")

    def _set_estimator(self) -> BaseRegressor:
        """Set regressor, split and late import due to tensorflow specifics

        Raises:
            NotImplementedError: regressor not supported

        Returns:
            BaseRegressor: derived regressor
        """

        if self.regressor_name == "CNNRegressor":
            from sktime.regression.deep_learning.cnn import CNNRegressor

            model = CNNRegressor
        elif self.regressor_name == "CNTCRegressor":
            from sktime.regression.deep_learning.cntc import CNTCRegressor

            model = CNTCRegressor
        elif self.regressor_name == "FCNRegressor":
            from sktime.regression.deep_learning.fcn import FCNRegressor

            model = FCNRegressor
        elif self.regressor_name == "LSTMFCNRegressor":
            from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor

            model = LSTMFCNRegressor
        elif self.regressor_name == "MLPRegressor":
            from sktime.regression.deep_learning.mlp import MLPRegressor

            model = MLPRegressor
        elif self.regressor_name == "ResNetRegressor":
            from sktime.regression.deep_learning.resnet import ResNetRegressor

            model = ResNetRegressor
        elif self.regressor_name == "InceptionTimeRegressor":
            from sktime.regression.deep_learning.inceptiontime import InceptionTimeRegressor

            model = InceptionTimeRegressor
        elif self.regressor_name == "MACNNRegressor":
            from sktime.regression.deep_learning.macnn import MACNNRegressor

            model = MACNNRegressor
        elif self.regressor_name == "MCDCNNRegressor":
            from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressor

            model = MCDCNNRegressor
        elif self.regressor_name == "SimpleRNNRegressor":
            from sktime.regression.deep_learning.rnn import SimpleRNNRegressor

            model = SimpleRNNRegressor
        elif self.regressor_name == "TapNetRegressor":
            from sktime.regression.deep_learning.tapnet import TapNetRegressor

            model = TapNetRegressor
        elif self.regressor_name == "KNeighborsTimeSeriesRegressor":
            from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor

            model = KNeighborsTimeSeriesRegressor
        elif self.regressor_name == "TimeSeriesForestRegressor":
            from sktime.regression.interval_based import TimeSeriesForestRegressor

            model = TimeSeriesForestRegressor
        elif self.regressor_name == "TimeSeriesSVRTslearn":
            from sktime.regression.kernel_based import TimeSeriesSVRTslearn

            model = TimeSeriesSVRTslearn
        elif self.regressor_name == "RocketRegressor":
            from sktime.regression.kernel_based import RocketRegressor

            model = RocketRegressor
        else:
            raise NotImplementedError(f"Regressor {self.regressor_name} not supported")

        valid_kwargs = inspect.signature(model).parameters.keys()
        self.hyperparameters = {k: v for k, v in self.hyperparameters.items() if k in valid_kwargs}
        return model(**self.hyperparameters)

    def predict(self, x: Data) -> Data:
        """Predict from the model

        Args:
            x (Data): data container with input data

        Returns:
            Data: data container with predicted data
        """
        # check
        self._check_fitted(x=x)

        # set data
        LOG.debug("Set data for prediction")
        x_in = self._prepare_input(x=x, training=False)
        if isinstance(x.get_tabular(), pd.DataFrame):
            idx = x.get_tabular().index
        else:
            idx = x.get_temporal().index.get_level_values(self._str.id).drop_duplicates()

        # predict
        LOG.debug("Start prediction")
        y_pred = self._model.predict(x_in)
        LOG.debug("Got prediction:\n%s", y_pred)

        if self.__test:
            LOG.critical("!! Test Mode !! - use Target from training to overwrite actual prediction!")
            y_pred = self.__y_store

        # transform back
        LOG.debug("Transform back")
        y_out = Data()

        if self._n_classes:
            LOG.debug("Assemble predictions from classification")
            y_pred = np.hstack(y_pred) if isinstance(y_pred, list) else y_pred
            if self._n_classes > 2:
                raise NotImplementedError("Multi-Class Classification not supported")
            else:
                raise NotImplementedError("Binary Classification not supported")
        elif self._is_regression:
            LOG.debug("Assemble regression prediction to DataFrame")
            y_out.set_tabular(pd.DataFrame(np.hstack(y_pred), index=idx, columns=self._label_names))
            LOG.debug("Transform back to original scale")
            y_out.set_tabular(self._transformer_target.inverse_transform(y_out.get_tabular()))
        else:
            raise NotImplementedError("Temporal target not supported")

        LOG.debug("Prediction done")

        return y_out

    def store(self) -> None:
        """Store the model"""
        LOG.info("Store model")
        self._model.save("model")

        if self._transformer_target is not None:
            with open("transformer_target.pkl", "wb") as f:
                pickle.dump(self._transformer_target, f)
        if self._transformer_feature_tabular is not None:
            with open("transformer_feature_tabular.pkl", "wb") as f:
                pickle.dump(self._transformer_feature_tabular, f)
        if self._transformer_feature_temporal is not None:
            with open("transformer_feature_temporal.pkl", "wb") as f:
                pickle.dump(self._transformer_feature_temporal, f)

        self._interface_names.store_params()

    def load(
        self,
        model_dir: Path,
        n_classes: int = 0,
        is_regression: bool = False,
        is_multi_channel_regression: bool = False,
        label_names: Optional[List[str]] = None,
        time_stamps: Optional[List[float]] = None,
    ) -> None:
        paras = json_util.load(model_dir / STR.fname_para)
        self.set_params(**paras[STR.pipeline])

        self._interface_names.load_params(inter_dir=model_dir)
        self._is_fitted = True

        if (model_dir / "transformer_target.pkl").is_file():
            with open(model_dir / "transformer_target.pkl", "rb") as f:
                self._transformer_target = pickle.load(f)
        if (model_dir / "transformer_feature_tabular.pkl").is_file():
            with open(model_dir / "transformer_feature_tabular.pkl", "rb") as f:
                self._transformer_feature_tabular = pickle.load(f)
        if (model_dir / "transformer_feature_temporal.pkl").is_file():
            with open(model_dir / "transformer_feature_temporal.pkl", "rb") as f:
                self._transformer_feature_temporal = pickle.load(f)

        self._model = BaseRegressor().load_from_path(model_dir / "model")

        self._n_classes = n_classes
        self._is_regression = is_regression
        self._is_multi_channel_regression = is_multi_channel_regression
        self._label_names = label_names
        self._time_stamps = time_stamps

    def _prepare_input(self, x: Data, training: bool = True) -> np.ndarray:
        """Set input data for the model

        Args:
            x (Data): feature data container

        Returns:
            List[np.ndarray]: one or two input arrays
        """
        x_in = []
        if isinstance(x.get_tabular(), pd.DataFrame):
            LOG.error("Tabular features not supported - IGNORE")

        if isinstance(x.get_temporal(), pd.DataFrame):
            LOG.debug("Transform temporal features")

            if training:
                self._transformer_feature_temporal = RobustScaler()
                self._transformer_feature_temporal.fit(x.get_temporal())

            x.set_temporal(self._transformer_feature_temporal.transform(x.get_temporal()))
            LOG.debug(
                "Robust Transformer with scale=%s and center=%s",
                self._transformer_feature_temporal.scale_,
                self._transformer_feature_temporal.center_,
            )
            x_in_, self.temporal_feature_n_tsps = x.get_temporal_3d(new_n_tsps=self.temporal_feature_n_tsps)
            x_in.append(x_in_)
        else:
            raise NotImplementedError("Pure Tabular features not supported")

        LOG.debug("Input shapes: %s", [q.shape for q in x_in])

        return x_in[0].transpose(0, 2, 1)

    def _prepare_output(self, y: Data) -> np.ndarray:
        """Prepare output data for the model

        Args:
            y (Data): target data container

        Returns:
            List[np.ndarray]: single output array
        """
        if isinstance(y.get_tabular(), pd.DataFrame):
            self._label_names = y.get_tabular().columns
            y_array = y.get_tabular().values
            y_vals = set(y_array.flatten())
            if y_vals == set([int(x) for x in y_vals]) and len(y_vals) > 2:
                raise NotImplementedError("Multi-Class Classification not supported")
            else:
                LOG.debug("Set target for Regression or Binary Classification")
                if len(y_vals) == 2:
                    raise NotImplementedError("Binary Classification not supported")
                else:
                    LOG.debug("Scale target for regression")
                    self._transformer_target = RobustScaler()
                    y.set_tabular(self._transformer_target.fit_transform(y.get_tabular()))
                    LOG.debug(
                        "Robust Transformer with scale=%s and center=%s",
                        self._transformer_target.scale_,
                        self._transformer_target.center_,
                    )
                    self._is_regression = True
                y_in = y.get_tabular().values
        else:
            raise NotImplementedError("Temporal target not supported")

        LOG.debug("Target shapes: %s", y_in.shape)

        return y_in


if __name__ == "__main__":
    # init
    init_logger(log_lvl=logging.DEBUG)

    # run
    LOG.critical("Start Checks")
    checker = CheckPipe(pipe=SktimeRegressor, n_samples=5)
    if False:
        LOG.critical("Generate Data")
        xx = Data()
        LOG.warning("Generate Features")
        xx.set_tabular(checker._get_tabular(n_columns=3, n_levels=np.inf))
        xx.set_temporal(checker._get_temporal(n_tsmps=3, n_channels=2))
        LOG.warning("Generate Target")
        yy = Data()
        # yy.set_tabular(checker._get_tabular(n_columns=3, n_levels=np.inf))
        yy.set_temporal(checker._get_temporal(n_tsmps=2, n_channels=3))
        LOG.critical("Data Generated")

        LOG.critical("Init Model")
        m = SktimeRegressor(test_mode=True)
        m.feature_extractor_path = Path("feature_extractor.weights.h5")
        LOG.critical("Model Initialized")
        LOG.critical("Fit Model")
        m.fit(x=xx, y=yy)
        LOG.critical("Model Fitted")
        LOG.critical("Predict")
        pred = m.predict(x=xx)
        LOG.debug(
            "Predicted Target Tabular %s:\n%s",
            None if pred.get_tabular() is None else pred.get_tabular().shape,
            pred.get_tabular(),
        )
        LOG.debug(
            "Predicted Target Temporal %s:\n%s",
            None if pred.get_temporal() is None else pred.get_temporal().shape,
            pred.get_temporal(),
        )
        m.store()
    else:
        checker.run_checks_all()
    LOG.critical("Checks Done")
