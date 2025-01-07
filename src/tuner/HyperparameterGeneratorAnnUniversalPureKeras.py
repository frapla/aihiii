import sys
import logging
from pathlib import Path
import numpy as np
import optuna

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator

LOG: logging.Logger = logging.getLogger(__name__)


class HyperparameterGeneratorAnnUniversalPureKeras(BaseHyperparameterGenerator):
    def __init__(self) -> None:
        super().__init__()

    def suggest_hyperparameter(self, trial: optuna.Trial) -> dict:
        """Draw new hyperparameter for each trial, suggestions have to be compatible with chosen sampler

        Args:
            trial (optuna.Trial): current trial

        Returns:
            dict: hyperparameter
        """
        # conv layers
        # [(n_filters, kernel_size), ...], outer list for horizontal and inner list for vertical stacking
        # conv_nfilters_and_size
        conv_width = trial.suggest_int("conv_width", low=1, high=4, step=1)
        conv_depth = trial.suggest_int("conv_depth", low=1, high=4, step=1)
        outer = []
        for width in range(conv_width):
            inner = []
            for depth in range(conv_depth):
                kernel_size = trial.suggest_int(f"kernel_size_{width}_{depth}", low=1, high=50, step=10)
                n_filters = trial.suggest_int(f"n_filters_{width}_{depth}", low=10, high=100, step=10)
                inner.append((n_filters, kernel_size))
            outer.append(inner)

        # dense layers
        n_dense_layers = trial.suggest_int("n_dense_layers", low=2, high=5, step=1)
        fst_dense_layer_shape = trial.suggest_int("fst_dense_layer_shape", low=10, high=100, step=10)
        last_dense_layer_shape = trial.suggest_int("last_dense_layer_shape", low=10, high=100, step=10)

        params = {
            "conv_nfilters_and_size": outer,
            "dense_layer_shapes": np.linspace(fst_dense_layer_shape, last_dense_layer_shape, n_dense_layers).astype(int),
            "pooling_size": trial.suggest_int("pooling_size", low=2, high=5, step=1),
            "pooling_strategy": trial.suggest_categorical("pooling_strategy", ["max", "average"]),
            "temporal_feature_n_tsps": trial.suggest_int("temporal_feature_n_tsps", low=60, high=180, step=40),
            "share_dense": trial.suggest_categorical("share_dense", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", low=1e-6, high=1e-1, log=True),
            "spatial_dropout_rate": trial.suggest_float("spatial_dropout_rate", low=0.0, high=0.5),
            "dense_regularizer": trial.suggest_categorical("dense_regularizer", [None, "l1", "l2"]),
        }

        return params
