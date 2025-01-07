import sys
import logging
from pathlib import Path
from datetime import datetime
import os
from itertools import product
from typing import List
import argparse

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.experiments._Experiment import Experiment
from src._StandardNames import StandardNames
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


def run(perc: int, tgt: str, database: str, ai_in: List[str]) -> None:
    cwd = os.getcwd()
    work_dir = (
        Path("experiments")
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{perc:02}HIII_{tgt}_from_{database}_ft_{'_'.join(ai_in)}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)

    exp = Experiment(
        user_pipeline=AnnUniversal,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=ai_in,
        file_names_ai_out=[tgt],
        feature_percentiles=[50],
        target_percentiles=[perc],
        used_columns_ai_out=[
            "Chest_Deflection",
        ],
        used_columns_ai_in=[
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
            "Head_HIC15",
            "Head_HIC36",
            "Head_a3ms",
            "Neck_Nij",
            "Neck_Fz_Max_Compression",
            "Neck_Fz_Max_Tension",
            "Neck_My_Max",
            "Neck_Fx_Shear_Max",
            "Chest_Deflection",
            "Chest_a3ms",
            "Chest_VC",
            "Femur_Fz_Max_Compression",
            "Femur_Fz_Max_Tension",
            "Femur_Fz_Max",
        ],
        hyperparameter={
            "conv_nfilters_and_size": [
                [[100, 41], [90, 41], [90, 11]],
                [[70, 41], [30, 1], [60, 41]],
                [[70, 11], [60, 11], [60, 11]],
            ],
            "dense_layer_shapes": [60, 60],
            "pooling_size": 5,
            "pooling_strategy": "average",
            "temporal_feature_n_tsps": 180,
            "share_dense": False,
            "learning_rate": 1.0268739540663257e-05,
            "spatial_dropout_rate": 0,
            "dense_regularizer": "l1",
            "used_temporal_features": [],
            "patience_factor": 0.01,
            "max_epochs": 3000,
            "start_early_stopping_from_n_epochs": 600,
            "baseline_threshold": 3,
            "feature_extractor_path": None,
        },
        shuffle_data=True,
        random_state_shuffle=42,
    )
    LOG.info("Prepare and run experiment")
    exp.prepare()
    exp.run()

    os.chdir(cwd)

    LOG.info("Done, back in %s", cwd)


def test() -> None:
    run(
        perc=5,
        tgt="injury_criteria_classes_Chest_Deflection_perc05_THRES_20_00_2",
        database="doe_sobol_20240705_194200",
        ai_in=["channels", "injury_criteria"],
    )


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.DEBUG)
    test()
