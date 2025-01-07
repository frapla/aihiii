import sys
import logging
from pathlib import Path
from datetime import datetime
import os
from itertools import product

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.BaseLineUniversal import BaseLineUniversal
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)


def run(perc: int, tgt: str, database: str) -> None:
    cwd = os.getcwd()
    work_dir = (
        Path("experiments")
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{perc:02}HIII_{tgt}_from_{database}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Working in %s", work_dir)

    exp = Experiment(
        user_pipeline=BaseLineUniversal,
        processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
        file_names_ai_in=["injury_criteria"],
        file_names_ai_out=[tgt],
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
        hyperparameter={},
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

    # run variation
    for database, perc, target in product(databases, percs, targets):
        run(perc=perc, tgt=target, database=database)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
