import sys
import logging
from pathlib import Path
from datetime import datetime
import os
from itertools import product
from typing import List, Union

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.PlainAnn import PlainAnn
from src.experiments._Experiment import Experiment
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)


def run(perc: Union[int, List[int]], tgt: str, database: str, ai_in: List[str]) -> None:
    cwd = os.getcwd()
    perc_name = f"{perc[0]:02}_{perc[1]:02}" if isinstance(perc, list) else f"{perc:02}"
    work_dir = (
        Path("experiments")
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{perc_name}HIII_{tgt}_from_{database}_ft_{'_'.join(ai_in)}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)

    exp = Experiment(
        user_pipeline=PlainAnn,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=ai_in,
        file_names_ai_out=[tgt],
        feature_percentiles=[50],
        target_percentiles=perc if isinstance(perc, list) else [perc],
        used_columns_ai_out=[
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
            "Head_a3ms",
            "Neck_My_Extension",
            "Neck_Fz_Max_Tension",
            "Neck_Fx_Shear_Max",
            "Chest_Deflection",
            "Femur_Fz_Max_Compression",
            "Chest_VC",
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
            "Neck_My_Extension",
            "Neck_My_Flexion",
            "Neck_Fx_Shear_Max",
            "Chest_Deflection",
            "Chest_a3ms",
            "Chest_VC",
            "Femur_Fz_Max_Compression",
            "Femur_Fz_Max_Tension",
            "Femur_Fz_Max",
        ],
        hyperparameter={
            "dense_layer_shapes": [60, 60],
            "temporal_feature_n_tsps": 70,
            "share_dense": False,
            "learning_rate": 1e-4,
            "dense_regularizer": "l1",
            "patience_factor": 0.01,
            "max_epochs": 3000,
            "start_early_stopping_from_n_epochs": 600,
            "baseline_threshold": 30,
            "plot_model": True,
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
    databases = ["doe_sobol_20240705_194200"]
    percs = [5, 95]
    targets = [
        f"injury_criteria{suffix}" if suffix != "channels" else suffix
        for suffix in ["", *[f"_classes_{i:d}" for i in [2, 3, 5, 7]], "channels"]
    ]
    targets = [targets[0]]
    ai_ins = [["channels", "injury_criteria"]]

    for database, perc, target, ai_in in product(databases, percs, targets, ai_ins):
        run(perc=perc, tgt=target, database=database, ai_in=ai_in)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
