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
from src.build.SktimeRegressor import SktimeRegressor
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)


def run(
    perc: Union[int, List[int]], database: str, regressor_name: str, used_column_ai_out: str, hyperparameters: dict = {}
) -> None:

    cwd = os.getcwd()
    perc_name = f"{perc[0]:02}_{perc[1]:02}" if isinstance(perc, list) else f"{perc:02}"
    work_dir = (
        Path("experiments")
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{regressor_name}_{perc_name}HIII_from_{database}_tgt_{used_column_ai_out}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)

    exp = Experiment(
        user_pipeline=SktimeRegressor,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=["channels"],
        file_names_ai_out=["injury_criteria"],
        feature_percentiles=[50],
        target_percentiles=perc if isinstance(perc, list) else [perc],
        used_columns_ai_out=[used_column_ai_out],
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
        ],
        hyperparameter={
            "regressor_name": regressor_name,
            "hyperparameters": hyperparameters,
            "temporal_feature_n_tsps": 140,
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
    used_columns_ai_out = [
        # "Head_HIC15",
        "Head_a3ms",
        # "Neck_My_Extension",
        # "Neck_Fz_Max_Tension",
        # "Neck_Fx_Shear_Max",
        "Chest_Deflection",
        # "Femur_Fz_Max_Compression",
        # "Chest_VC",
    ]
    regressor_names = SktimeRegressor().get_regressor_names()

    for database, regressor_name, perc, used_column_ai_out in product(databases, regressor_names, percs, used_columns_ai_out):
        if (perc == 5 and regressor_name == regressor_names[0]) or regressor_name == regressor_names[1]:
            continue
        run(
            perc=perc,
            database=database,
            regressor_name=regressor_name,
            used_column_ai_out=used_column_ai_out,
            hyperparameters={"verbose": True},
        )


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
