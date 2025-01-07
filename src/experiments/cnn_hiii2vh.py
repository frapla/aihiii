import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src._StandardNames import StandardNames
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)


def run(
    feature_percentiles: Union[int, List[int]],
    target_percentiles: Union[int, List[int]],
    tgt: str,
    database: str,
    ai_in: List[str],
) -> None:
    cwd = os.getcwd()
    tg_perc_name = (
        f"{target_percentiles[0]:02}_{target_percentiles[1]:02}"
        if isinstance(target_percentiles, list)
        else f"{target_percentiles:02}"
    )
    ft_perc_name = (
        f"{feature_percentiles[0]:02}_{feature_percentiles[1]:02}"
        if isinstance(feature_percentiles, list)
        else f"{feature_percentiles:02}"
    )
    work_dir = (
        Path("experiments")
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_from_{database}_ft_{'_'.join(ai_in)}_in_{ft_perc_name}_tg_{tg_perc_name}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)

    exp = Experiment(
        user_pipeline=AnnUniversal,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=ai_in,
        file_names_ai_out=[tgt],
        feature_percentiles=feature_percentiles if isinstance(feature_percentiles, list) else [feature_percentiles],
        target_percentiles=target_percentiles if isinstance(target_percentiles, list) else [target_percentiles],
        used_columns_ai_out=[
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
    databases = ["unite_hiii_virthuman"]
    perc_ft_tgs = [[5, 6], [50, 51], [95, 96], [50, 6], [50, 96]]
    targets = ["injury_criteria"]

    ai_ins = [["channels"]]

    for database, perc_ft_tg, target, ai_in in product(databases, perc_ft_tgs, targets, ai_ins):
        run(feature_percentiles=perc_ft_tg[0], target_percentiles=perc_ft_tg[1], tgt=target, database=database, ai_in=ai_in)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
