from typing import Optional, List, Union, Literal
import keras
import logging
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import numpy as np
import sys
import json
from pathlib import Path
import pickle
import pandas as pd
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Concatenate,
    Layer,
    AveragePooling1D,
    SpatialDropout1D,
)
from keras import backend as K

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src.build.CheckPipe import CheckPipe
from src.utils.custom_log import init_logger
import src.utils.json_util as json_util
from src.build.EarlyStoppingBaselineCallback import EarlyStoppingBaseline
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class AnnUniversal(BasePipe):
    def __init__(self, test_mode: bool = False) -> None:
        """Universal ANN with Keras"""
        # init
        super().__init__()

        # parameter
        # [(n_filters, kernel_size), ...], outer list for horizontal and inner list for vertical stacking
        self.conv_nfilters_and_size = [[(100, 100), (50, 100)], [(25, 10)]]
        self.dense_layer_shapes: List[int] = [60, 10]
        self.pooling_size: int = 2
        self.pooling_strategy: Literal["max", "average"] = "max"
        self.temporal_feature_n_tsps: int = 140
        self.share_dense: bool = False
        self.learning_rate: float = 0.001
        self.spatial_dropout_rate: float = 0
        self.dense_regularizer: Optional[Literal["l1", "l2"]] = None
        self.patience_factor: float = 0.1
        self.max_epochs: int = 1000
        self.start_early_stopping_from_n_epochs: int = 500
        self.baseline_threshold: Optional[int] = 3
        self.feature_extractor_path: Optional[Path] = None
        self.n_epochs_fine_tuning: Optional[int] = None
        self.fine_tuning_lr_factor: float = 0.1
        self.plot_model: bool = True

        # misc
        self._channel_format: str = "channels_last"
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
        self._feature_extractor: Optional[keras.Model] = None
        self.__history: Optional[keras.callbacks.History] = None
        self._early_stops: Optional[List[keras.callbacks.EarlyStopping]] = None

    def _set_early_stop(self) -> None:
        self._early_stops: List[keras.callbacks.EarlyStopping] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.patience_factor * self.max_epochs),
                start_from_epoch=self.start_early_stopping_from_n_epochs // 2,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        if self.baseline_threshold is not None:
            self._early_stops.append(
                EarlyStoppingBaseline(
                    monitor="val_loss",
                    restore_best_weights=False,
                    patience=int(self.patience_factor * self.max_epochs),
                    start_from_epoch=self.start_early_stopping_from_n_epochs,
                    baseline=self.baseline_threshold,
                    verbose=1,
                )
            )

    def get_params(self) -> dict:
        """Read the parameters of the model

        Returns:
            dict: parameters
        """
        early_stop_params = []
        for early_stop in self._early_stops:
            early_stop_params.append({})
            for key, val in early_stop.__dict__.items():
                try:
                    json.dumps(val)
                    early_stop_params[-1][key] = val
                except TypeError:
                    pass

        h_para = {
            "set_params": {p: v for p, v in self.__dict__.items() if not p.startswith("_")},
            "model": self._model.get_config(),
            "optimizer": self._model.optimizer.get_config(),
            "earlystop": early_stop_params,
            "loss": self._model.loss,
            "training_params": self._model.history.params,
            "training_history": {"train": self.__history.history["loss"], "val": self.__history.history["val_loss"]},
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
        if isinstance(y.get_temporal(), pd.DataFrame):
            self._interface_names.orig_target_tsp = len(y.get_temporal().index.get_level_values(STR.time).unique())

        x_in = self._prepare_input(x=x, training=True)
        y_in = self._prepare_output(y=y)

        for i, x_in_i in enumerate(x_in):
            LOG.debug("Pre-Processed Input %s %s:\n%s", i, x_in_i.shape, x_in_i)
        for i, y_in_i in enumerate(y_in):
            LOG.debug("Pre-Processed Target %s %s:\n%s", i, y_in_i.shape, y_in_i)

        # set model
        LOG.debug("Build model")
        self._build(x=x, y=y)

        # training
        LOG.debug("Start training")
        self._set_early_stop()
        self._model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self._loss)
        self.__history = self._model.fit(
            x_in,
            y_in,
            validation_split=0.1,
            callbacks=self._early_stops,
            epochs=self.max_epochs,
        )
        if self.__test:
            self.__y_store = y_in

        if self.feature_extractor_path is not None and self.n_epochs_fine_tuning is not None:
            LOG.info("Unfreeze Feature Extractor Layers")
            self._feature_extractor.trainable = True
            self._model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate * self.fine_tuning_lr_factor), loss=self._loss
            )
            self._model.fit(
                x_in,
                y_in,
                validation_split=0.1,
                callbacks=self._early_stops,
                epochs=self.n_epochs_fine_tuning,
            )
        LOG.debug("Training done")

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
                LOG.debug("Reverse OneHotEncoder")
                y_pred = self._transformer_target.inverse_transform(y_pred)
            else:
                LOG.debug("Convert probabilities to classes")
                y_pred = (y_pred > 0.5).astype(int)
            LOG.debug("Convert prediction to DataFrame")
            y_out.set_tabular(pd.DataFrame(y_pred, index=idx, columns=self._label_names))
        elif self._is_regression:
            LOG.debug("Assemble regression prediction to DataFrame")
            y_out.set_tabular(pd.DataFrame(np.hstack(y_pred), index=idx, columns=self._label_names))
            LOG.debug("Transform back to original scale")
            y_out.set_tabular(self._transformer_target.inverse_transform(y_out.get_tabular()))
        else:
            LOG.debug("Assemble temporal prediction to DataFrame")
            y_out.set_temporal_3d(
                data=np.dstack(y_pred) if isinstance(y_pred, list) else y_pred,
                time_stmps=self._time_stamps,
                idx=idx,
                columns=self._label_names,
            )
            LOG.debug("Transform back to original scale")
            y_out.set_temporal(self._transformer_target.inverse_transform(y_out.get_temporal()))

        LOG.debug("Prediction done")

        return y_out

    def store(self) -> None:
        """Store the model"""
        LOG.info("Store model")
        self._model.save("model.keras")
        if self._feature_extractor is not None:
            self._feature_extractor.save_weights("feature_extractor.weights.h5")

        if self.plot_model:
            try:
                keras.utils.plot_model(
                    self._model,
                    to_file="model.png",
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir="TB",
                    expand_nested=True,
                    dpi=200,
                    show_layer_activations=True,
                    show_trainable=True,
                )
            except:
                LOG.error("Could not plot model")

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

        self._model = keras.models.load_model(model_dir / "model.keras")

        self._n_classes = n_classes
        self._is_regression = is_regression
        self._is_multi_channel_regression = is_multi_channel_regression
        self._label_names = label_names
        self._time_stamps = time_stamps

    def _prepare_input(self, x: Data, training: bool = True) -> List[np.ndarray]:
        """Set input data for the model

        Args:
            x (Data): feature data container

        Returns:
            List[np.ndarray]: one or two input arrays
        """
        x_in = []
        if isinstance(x.get_tabular(), pd.DataFrame):
            LOG.debug("Transform tabular features")

            if training:
                self._transformer_feature_tabular = RobustScaler()
                self._transformer_feature_tabular.fit(x.get_tabular())
            x_in.append(self._transformer_feature_tabular.transform(x.get_tabular()))
            LOG.debug(
                "Robust Transformer with scale=%s and center=%s",
                self._transformer_feature_tabular.scale_,
                self._transformer_feature_tabular.center_,
            )
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

        LOG.debug("Input shapes: %s", [q.shape for q in x_in])

        return x_in

    def _prepare_output(self, y: Data) -> List[np.ndarray]:
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
                LOG.debug("Set target for Multi-Class Classification (OneHotEncoder)")
                self._n_classes = len(y_vals)
                self._transformer_target = OneHotEncoder(
                    sparse_output=False,
                    categories=[sorted(y_vals)] * y_array.shape[1],
                )
                y_in = self._transformer_target.fit_transform(y.get_tabular())
                y_in = np.hsplit(y_in, y_array.shape[1])
            else:
                LOG.debug("Set target for Regression or Binary Classification")
                if len(y_vals) == 2:
                    self._n_classes = len(y_vals)
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
                y_in = [y.get_tabular()[[c]].to_numpy() for c in y.get_tabular().columns]
        else:
            LOG.debug("Set target for Multi Channel Regression")
            self._label_names = y.get_temporal().columns
            self._time_stamps = y.get_temporal().index.get_level_values(self._str.time).drop_duplicates()
            self._transformer_target = RobustScaler()
            y.set_temporal(self._transformer_target.fit_transform(y.get_temporal()))
            LOG.debug(
                "Robust Transformer with scale=%s and center=%s",
                self._transformer_target.scale_,
                self._transformer_target.center_,
            )
            y_, self.temporal_feature_n_tsps = y.get_temporal_3d(new_n_tsps=self.temporal_feature_n_tsps)
            y_in = np.dsplit(y_, y.get_temporal().shape[1])
            self._is_multi_channel_regression = True

        LOG.debug("Target shapes: %s", [q.shape for q in y_in])

        return y_in

    def _build(self, x: Data, y: Data) -> None:
        """Build model

        Args:
            x (Data): feature data container
            y (Data): target data container
        """
        # tabular input
        LOG.info("Set tabular input")
        if isinstance(x.get_tabular(), pd.DataFrame):
            tab_in = self._tabular_input(x=x.get_tabular())
        else:
            tab_in = None
        LOG.debug("Tabular Input: %s", tab_in)

        # temporal input
        LOG.info("Set temporal input")
        if isinstance(x.get_temporal(), pd.DataFrame):
            # input
            temp_in = self._temporal_input(x=x.get_temporal_resampled(new_n_tsps=self.temporal_feature_n_tsps))

            # feature extractor
            self._set_feature_extractor(temp_in=temp_in)
            feature_extractor_block = self._feature_extractor(temp_in)
        else:
            temp_in, feature_extractor_block = None, None
        LOG.debug("Temporal Input: %s and feature extractor %s", temp_in, feature_extractor_block)

        # inputs
        LOG.info("Set input layers")
        if tab_in is not None and temp_in is not None:
            inputs = [tab_in, temp_in]
        elif tab_in is not None:
            inputs = [tab_in]
        else:
            inputs = [temp_in]
        LOG.debug("Input Layers: %s", inputs)

        # concatenate
        LOG.info("Set dense input")
        if tab_in is None:
            dense_block = feature_extractor_block
        elif feature_extractor_block is None:
            dense_block = tab_in
        else:
            dense_block = self._concatenate()([tab_in, feature_extractor_block])
        LOG.debug("Concatenated Input: %s", dense_block)

        # dense
        LOG.info("Set dense layers")
        if self.share_dense:
            for dense_layer in self._dense():
                dense_block = dense_layer(dense_block)
        else:
            dense_stack = []
            for dense_vert in self._dense():
                dense_stack.append(dense_block)
                for dense_layer in dense_vert:
                    dense_stack[-1] = dense_layer(dense_stack[-1])
        LOG.debug("Dense Block: %s", dense_block)

        # output
        LOG.info("Set output layers")
        out_block = []
        for i, output_layer in enumerate(self._output(y=y)):
            out_block.append(output_layer(dense_block if self.share_dense else dense_stack[i]))
        LOG.debug("Output Block: %s", out_block)

        # compile
        LOG.info("Compile model with loss: %s", self._loss)
        self._model: keras.Model = keras.Model(inputs=inputs, outputs=out_block)
        LOG.debug("Model assembled: %s", self._model)

    def _set_feature_extractor(self, temp_in: Input) -> None:
        LOG.debug("Assemble feature extractor")
        feature_extractor_in = Input(shape=temp_in.shape[1:], name="FeatureExtractorIn")
        LOG.debug("Add Feature Extractor Input Layer: %s", feature_extractor_in)
        conv_block = []
        for conv_vert, pool_vert, dropout_vert in zip(self._convolutional_layers(), self._max_pooling(), self._spatial_dropout()):
            conv_block.append(feature_extractor_in)
            for conv, pool, dropout in zip(conv_vert, pool_vert, dropout_vert):
                conv_block[-1] = conv(conv_block[-1])
                if self.spatial_dropout_rate:
                    conv_block[-1] = dropout(conv_block[-1])
                conv_block[-1] = pool(conv_block[-1])
            conv_block[-1] = self._flatten()(conv_block[-1])
        if len(conv_block) > 1:
            temp_block = self._concatenate()(conv_block)
        else:
            temp_block = conv_block[0]

        self._feature_extractor: keras.Model = keras.Model(
            inputs=feature_extractor_in, outputs=temp_block, name=self._str.feature_extractor
        )
        LOG.debug("Feature Extractor assembled: %s", self._feature_extractor)
        if self.feature_extractor_path is not None:
            LOG.info("Load Feature Extractor Weights from %s", self.feature_extractor_path)
            self._feature_extractor.load_weights(filepath=self.feature_extractor_path)
            LOG.info("Freeze Feature Extractor Layers")
            self._feature_extractor.trainable = False

    def _tabular_input(self, x: pd.DataFrame) -> Layer:
        """Set tabular input

        Args:
            x (pd.DataFrame): data container

        Returns:
            Layer: input layer
        """
        inp = Input(shape=(x.shape[1],), name=f"TabularIn{len(x.index.names)}")
        LOG.debug("Add Tabular Input Layer: %s", inp)

        return inp

    def _temporal_input(self, x: pd.DataFrame) -> Layer:
        """temporal input

        Args:
            x (pd.DataFrame): data container

        Returns:
            Layer: input layer
        """
        n_tsps = (
            x.index.get_level_values("TIME").nunique() if self.temporal_feature_n_tsps is None else self.temporal_feature_n_tsps
        )
        inp = Input(shape=(n_tsps, x.shape[1]), name="TemporalIn")
        LOG.debug("Add Temporal Input Layer: %s", inp)
        return inp

    def _convolutional_layers(self) -> List[List[Layer]]:
        """convolutional 1D

        Returns:
            List[Layer]: convolutional layers
        """
        convs = []
        for horizonal in self.conv_nfilters_and_size:
            conv_vert = []
            for n_filters, kernel_size in horizonal:
                conv = Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                    data_format=self._channel_format,
                )
                conv_vert.append(conv)
            convs.append(conv_vert)
            LOG.debug("Add Convolutional Layer %s", convs[-1])

        return convs

    def _max_pooling(self) -> List[List[Layer]]:
        """MaxPooling1D

        Returns:
            List[Layer]: max pooling layers
        """
        pools = []
        for horizonal in self.conv_nfilters_and_size:
            pools_vert = []
            for _ in horizonal:
                pool_strat = MaxPooling1D if self.pooling_strategy == "max" else AveragePooling1D
                pool = pool_strat(
                    pool_size=self.pooling_size,
                    strides=None,
                    padding="same",
                    data_format=self._channel_format,
                )
                pools_vert.append(pool)
                LOG.debug("Add Pooling Layer %s", pools_vert[-1])
            pools.append(pools_vert)
        return pools

    def _spatial_dropout(self) -> List[List[Layer]]:
        pools = []
        for horizonal in self.conv_nfilters_and_size:
            pools_vert = []
            for _ in horizonal:
                pools_vert.append(SpatialDropout1D(rate=self.spatial_dropout_rate))
                LOG.debug("Add Spatial Dropout Layer %s", pools_vert[-1])
            pools.append(pools_vert)
        return pools

    def _flatten(self) -> Layer:
        """flatten

        Returns:
            Layer: flatten layer
        """
        fl = Flatten(data_format=self._channel_format)
        LOG.debug("Add Flatten Layer %s", fl)
        return fl

    def _concatenate(self) -> Layer:
        """Concatenate

        Returns:
            Layer: concatenate layer
        """
        conc = Concatenate()
        LOG.debug("Add Concatenate Layer %s", conc)
        return conc

    def _dense(self) -> Union[List[Layer], List[List[Layer]]]:
        """Define dense layers

        Returns:
            List[Layer]: dense layers
        """
        if self.share_dense:
            dense = []
            for dense_layer_shape in self.dense_layer_shapes:
                dense.append(
                    Dense(
                        dense_layer_shape,
                        activation="relu",
                        kernel_regularizer=self.dense_regularizer,
                    )
                )
                LOG.debug("Add Dense Layer %s", dense[-1])

            return dense
        else:
            dense = []
            for _ in self._label_names:
                dense_vert = []
                for dense_layer_shape in self.dense_layer_shapes:
                    dense_vert.append(
                        Dense(
                            dense_layer_shape,
                            activation="relu",
                            kernel_regularizer=self.dense_regularizer,
                        )
                    )
                    LOG.debug("Add Dense Layer %s", dense_vert[-1])
                dense.append(dense_vert)

            return dense

    def _output(self, y: Data) -> List[Dense]:
        """Define output layers

        Args:
            y (Data): target data container

        Raises:
            ValueError: unknown/mixed target type

        Returns:
            List[Dense]: output layers
        """
        if self._n_classes:
            return self._classification(y.get_tabular())
        elif self._is_regression:
            return self._regression(y.get_tabular())
        elif self._is_multi_channel_regression:
            return self._regression(y.get_temporal())
        else:
            raise ValueError("Unknown Target Type")

    def _classification(self, y: pd.DataFrame) -> List[Dense]:
        """Classification output

        Args:
            y (pd.DataFrame): target data container

        Returns:
            List[Dense]: dense output layers for classification
        """
        outs = []
        loss = []
        for i, col in enumerate(y.columns):
            out = Dense(
                1 if self._n_classes == 2 else self._n_classes,
                activation="sigmoid" if self._n_classes == 2 else "softmax",
                name=f"Classification_{i}_{col}",
                kernel_regularizer=self.dense_regularizer,
            )
            outs.append(out)
            loss.append("binary_crossentropy" if self._n_classes == 2 else "categorical_crossentropy")
            LOG.debug("Add Classification Output Layer %s %s", i, out)

        self._loss = loss
        return outs

    def _regression(self, y: pd.DataFrame) -> List[Dense]:
        """Regression output

        Args:
            y (pd.DataFrame): target data container

        Returns:
            List[Dense]: output layers for regression
        """
        outs = []
        for i, col in enumerate(y.columns):
            out = Dense(
                1 if len(y.index.names) == 1 else self.temporal_feature_n_tsps,
                activation="linear",
                name=f"RegressionOut_{i}_{col}",
                kernel_regularizer=self.dense_regularizer,
            )
            LOG.debug("Add Regression Output Layer %s %s", i, out)
            outs.append(out)
        self._loss = ["mae"] * len(y.columns)

        return outs


if __name__ == "__main__":
    # init
    init_logger(log_lvl=logging.DEBUG)

    # run
    LOG.critical("Start Checks")
    checker = CheckPipe(pipe=AnnUniversal, n_samples=5)
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
        m = AnnUniversal(test_mode=True)
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
