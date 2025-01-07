import sys
import logging
from pathlib import Path
from datetime import datetime
import os
from itertools import product
from typing import List, Union, Optional, Literal

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.RocketAnn import RocketAnn
from src.experiments._Experiment import Experiment
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)


def run(
    perc: Union[int, List[int]],
    tgt: str,
    database: str,
    ai_in: List[str],
    n_kernels: int = 40,
) -> None:
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
        user_pipeline=RocketAnn,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=ai_in,
        file_names_ai_out=[tgt],
        feature_percentiles=[50],
        target_percentiles=perc if isinstance(perc, list) else [perc],
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
            "n_kernels": n_kernels,
            "dense_layer_shapes": [100, 96, 93, 90],
            "share_dense": False,
            "learning_rate": 1e-5,
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
    databases = ["doe_sobol_20240705_194200"]
    percs = [95]  # , 95]
    targets = ["injury_criteria"]
    ai_ins = [["channels"]]
    nkks = [20, 50, 100, 1000]

    for database, perc, target, ai_in, n_ks in product(databases, percs, targets, ai_ins, nkks):
        run(perc=perc, tgt=target, database=database, ai_in=ai_in, n_kernels=n_ks)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
