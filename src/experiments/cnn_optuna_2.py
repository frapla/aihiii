import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

import optuna
import optuna.visualization as ov

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator
from src.tuner._Study import Study
from src.tuner.Hyperparameter import Hyperparameter
from src.utils.custom_log import init_logger

LOG: logging.Logger = logging.getLogger(__name__)


class HyperGenerator(BaseHyperparameterGenerator):
    def __init__(self, target_percentile: int, database: str) -> None:
        super().__init__()
        self.target_percentile = target_percentile
        self.database = database

    def suggest_hyperparameter(self, trial: optuna.Trial) -> Hyperparameter:
        """Draw new hyperparameter for each trial, suggestions have to be compatible with chosen sampler

        Args:
            trial (optuna.Trial): current trial

        Returns:
            dict: hyperparameter
        """
        file_names_ai_in = trial.suggest_categorical(
            "file_names_ai_in",
            [
                "channels",
                "channels_injury",
                "injury",
            ],
        )
        used_columns_ai_in = []

        temporal_channels = [
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03HEAD0000OCCUACXD",
            "03HEAD0000OCCUACYD",
            "03HEAD0000OCCUACZD",
            "03CHSTLOC0OCCUDSXD",
            "03CHSTLOC0OCCUDSYD",
            "03CHSTLOC0OCCUDSZD",
            "03CHST0000OCCUDSXD",
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACYD",
            "03CHST0000OCCUACZD",
            "03PELVLOC0OCCUDSXD",
            "03PELVLOC0OCCUDSYD",
            "03PELVLOC0OCCUDSZD",
            "03PELV0000OCCUACXD",
            "03PELV0000OCCUACYD",
            "03PELV0000OCCUACZD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03FEMRRI00OCCUFOZD",
            "03FEMRLE00OCCUFOZD",
        ]
        inj_vals = [
            "Head_HIC15",
            "Head_a3ms",
            "Neck_Nij",
            "Neck_Fz_Max_Compression",
            "Neck_Fz_Max_Tension",
            "Neck_My_Extension",
            "Neck_My_Flexion",
            "Neck_Fx_Shear_Max",
            "Chest_Deflection",
            "Chest_a3ms",
            "Chest_VC",
            "Femur_Fz_Max_Compression",
            "Femur_Fz_Max_Tension",
        ]

        # conv layers
        # [(n_filters, kernel_size), ...], outer list for horizontal and inner list for vertical stacking
        # conv_nfilters_and_size
        if "channels" in file_names_ai_in:
            for channel in temporal_channels:
                if trial.suggest_categorical(channel, [True, False]):
                    used_columns_ai_in.append(channel)
            if len(used_columns_ai_in) == 0:
                used_columns_ai_in.append(temporal_channels[0])

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

            dropout_conv = trial.suggest_categorical("dropout_conv", [True, False])

            pooling_size = trial.suggest_int("pooling_size", low=2, high=5, step=1)
            pooling_strategy = trial.suggest_categorical("pooling_strategy", ["max", "average"])
            temporal_feature_n_tsps = trial.suggest_categorical("temporal_feature_n_tsps", [70, 140, 1400])

        else:
            outer = None
            dropout_conv = None
            pooling_size = None
            pooling_strategy = None
            temporal_feature_n_tsps = None

        if "injury" in file_names_ai_in:
            tmp = []
            for inj in inj_vals:
                if trial.suggest_categorical(inj, [True, False]):
                    tmp.append(inj)
            if len(tmp) == 0:
                tmp.append(inj_vals[0])
            used_columns_ai_in.extend(tmp)

        # dense layers
        n_dense_layers = trial.suggest_int("n_dense_layers", low=2, high=5, step=1)
        fst_dense_layer_shape = trial.suggest_int("fst_dense_layer_shape", low=10, high=100, step=10)
        last_dense_layer_shape = trial.suggest_int("last_dense_layer_shape", low=10, high=100, step=10)
        spatial_dropout_rate = trial.suggest_float("spatial_dropout_rate", low=0.1, high=0.5) if dropout_conv else 0

        params = Hyperparameter()
        # pipeline
        params.database = self.database
        params.target_percentiles = [self.target_percentile]
        params.file_names_ai_out = ["injury_criteria"]

        if file_names_ai_in == "channels":
            params.file_names_ai_in = ["channels"]
        elif file_names_ai_in == "channels_injury":
            params.file_names_ai_in = ["channels", "injury_criteria"]
        elif file_names_ai_in == "injury":
            params.file_names_ai_in = ["injury_criteria"]

        params.used_columns_ai_out = [
            "Head_HIC15",
            "Head_a3ms",
            "Chest_a3ms",
            "Neck_My_Extension",
            "Neck_Fz_Max_Tension",
            "Neck_Fx_Shear_Max",
            "Chest_Deflection",
            "Femur_Fz_Max_Compression",
            "Chest_VC",
        ]
        params.used_columns_ai_in = used_columns_ai_in

        # estimator
        params.estimator_hyperparameter = {
            "conv_nfilters_and_size": outer,
            "dense_layer_shapes": np.linspace(fst_dense_layer_shape, last_dense_layer_shape, n_dense_layers).astype(int),
            "pooling_size": pooling_size,
            "pooling_strategy": pooling_strategy,
            "temporal_feature_n_tsps": temporal_feature_n_tsps,
            "share_dense": bool(trial.suggest_categorical("share_dense", [1, 0])),
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 1e-4, 1e-5]),
            "spatial_dropout_rate": spatial_dropout_rate,
            "dense_regularizer": trial.suggest_categorical("dense_regularizer", [None, "l1", "l2"]),
            "patience_factor": 0.01,
            "max_epochs": 3000,
            "start_early_stopping_from_n_epochs": 600,
            "baseline_threshold": 3,
            "feature_extractor_path": None,
        }

        return params


def evaluate_study(study: optuna.Study):
    """Simple example of accessing study results
       Alternative to optuna dashboard (VSCode extension)

    Args:
        study (optuna.Study): filled study object
    """
    print(study.trials_dataframe())
    try:
        ov.plot_param_importances(study=study).show()
    except RuntimeError as er:
        LOG.warning("Data to simple for plot - %s - SKIP plotting", er)


def test():
    optuna.logging.disable_default_handler()
    database = "doe_sobol_20240705_194200"

    cwd = os.getcwd()
    work_dir = Path("experiments") / f"2024-11-10-21-13-00_pure_cnn_optuna_{database}"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    # init
    study = Study(
        user_pipeline=AnnUniversal,
        hyperparameter_generator=HyperGenerator(target_percentile=95, database=database),
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        n_trials=200,
        n_jobs=1,  # only threading
        study_name="AnnUniversalPureKeras",
    )

    # run
    results = study.run_study()

    # evaluate
    evaluate_study(study=results)
    os.chdir(cwd)
    LOG.info("Done")


if __name__ == "__main__":
    init_logger(log_lvl=logging.INFO)
    test()
