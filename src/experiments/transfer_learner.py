import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional

import pandas as pd

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)


def run(perc: int, doe: str, feature_extractor_path: Optional[Path] = None, n_epochs_fine_tuning: Optional[int] = 3000) -> None:
    cwd = os.getcwd()
    work_dir = Path("experiments") / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{perc:02}HIII_{doe}"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)
    exp = Experiment(
        user_pipeline=AnnUniversal,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / doe,
        file_names_ai_in=["channels"],
        file_names_ai_out=["injury_criteria"],
        feature_percentiles=[50],
        target_percentiles=[perc],
        used_columns_ai_out=[
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03HEAD0000OCCUACXD",
            "03HEAD0000OCCUACYD",
            "03HEAD0000OCCUACZD",
            "03HEAD0000OCCUACRD",
            "03CHSTLOC0OCCUDSXD",
            "03CHSTLOC0OCCUDSYD",
            "03CHSTLOC0OCCUDSZD",
            "03CHST0000OCCUDSXD",
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACYD",
            "03CHST0000OCCUACZD",
            "03CHST0000OCCUACRD",
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
            "Chest_Deflection",
            "Chest_VC",
            "Chest_a3ms",
            "Femur_Fz_Max_Compression",
            "Head_HIC15",
            "Head_a3ms",
            "Neck_Fx_Shear_Max",
            "Neck_Fz_Max_Tension",
            "Neck_My_Extension",
        ],
        used_columns_ai_in=[
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACZD",
            "03CHST0000OCCUDSXD",
            "03CHSTLOC0OCCUDSXD",
            "03HEAD0000OCCUACZD",
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03PELV0000OCCUACYD",
            "03PELV0000OCCUACZD",
            "03PELVLOC0OCCUDSXD",
            "03PELVLOC0OCCUDSYD",
            "03PELVLOC0OCCUDSZD",
        ],
        hyperparameter={
            "conv_nfilters_and_size": [[[60, 41]], [[90, 41]], [[50, 11]]],
            "dense_layer_shapes": [100, 96, 93, 90],
            "pooling_size": 5,
            "pooling_strategy": "average",
            "temporal_feature_n_tsps": 70,
            "share_dense": False,
            "learning_rate": 1e-5,
            "spatial_dropout_rate": 0.33269266283466437,
            "dense_regularizer": "l2",
            "patience_factor": 0.01,
            "max_epochs": 3000,
            "start_early_stopping_from_n_epochs": 600,
            "baseline_threshold": 30,
            "feature_extractor_path": feature_extractor_path,
            "n_epochs_fine_tuning": n_epochs_fine_tuning,
        },
        shuffle_data=True,
        random_state_shuffle=42,
    )
    LOG.info("Prepare and run experiment")
    exp.prepare()
    exp.run()

    os.chdir(cwd)

    LOG.info("Done, back in %s", cwd)

    return work_dir


def test() -> None:
    percs = [
        5,
        95,
    ]
    pre_trained = {}
    if True:

        for perc, doe in product(percs, ["doe_sobol_20240705_194200", "virthuman"]):
            w_dir = run(perc=perc, doe=doe)
            if doe != "virthuman":
                pre_trained[perc] = w_dir / "feature_extractor.weights.h5"
    else:
        pre_trained = {
            95: Path(
                "experiments/2024-11-13-22-40-31_transfer_learner_95HIII_doe_sobol_20240705_194200/feature_extractor.weights.h5"
            ),
            5: Path(
                "experiments/2024-11-13-20-56-43_transfer_learner_05HIII_doe_sobol_20240705_194200/feature_extractor.weights.h5"
            ),
        }
    for perc, feature_extr in pre_trained.items():
        run(perc=perc, doe="virthuman", feature_extractor_path=feature_extr.absolute(), n_epochs_fine_tuning=3000)
        run(perc=perc, doe="virthuman", feature_extractor_path=feature_extr.absolute(), n_epochs_fine_tuning=None)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
