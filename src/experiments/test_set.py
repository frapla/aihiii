import argparse
import logging
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.evaluate._Data import Data
from src.evaluate._Metrics import Metrics
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()

USED_COLUMNS_AI_IN: List[str] = [
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
]

USED_COLUMNS_AI_OUT: List[str] = [
    "Chest_Deflection",
    "Chest_VC",
    "Chest_a3ms",
    "Femur_Fz_Max_Compression",
    "Head_HIC15",
    "Head_a3ms",
    "Neck_Fx_Shear_Max",
    "Neck_Fz_Max_Tension",
    "Neck_My_Extension",
]


def run(
    feature_percentiles: Union[int, List[int]],
    target_percentiles: Union[int, List[int]],
    tgt: str,
    database: str,
    ai_in: List[str],
) -> Path:
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
        used_columns_ai_out=USED_COLUMNS_AI_OUT,
        used_columns_ai_in=USED_COLUMNS_AI_IN,
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


def do_prediction(model_dir: Path, tgt_perc: int = 95, database: str = "doe_sobol_test_20240829_135200") -> None:
    ft_perc = 50
    # init book
    book = {STR.creation: datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}

    # directory setup
    cwd = os.getcwd()
    work_dir = (
        Path("experiments") / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_predictions_{tgt_perc}HIII"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    d_dir = Path("..") / ".." / "data" / "doe" / database

    # get ids
    drops: Dict[int, List[int]] = {int(p): vs for p, vs in json_util.load(f_path=d_dir / STR.fname_dropped_ids).items()}
    existing_ids = []
    for perc, db_name in ((ft_perc, STR.fname_channels), (ft_perc, STR.fname_injury_crit), (tgt_perc, STR.fname_injury_crit)):
        existing_ids.append(
            set(
                pl.scan_parquet(d_dir / db_name)
                .filter(pl.col(STR.perc) == perc)
                .select(pl.col(STR.id).cast(pl.Int32))
                .filter(~pl.col(STR.id).is_in(drops[perc]))
                .unique()
                .collect()[STR.id]
                .to_list()
            )
        )
    sel_ids = sorted(existing_ids[0] & existing_ids[1] & existing_ids[2])

    # load model
    cnn = AnnUniversal()
    cnn.load(model_dir=model_dir, is_regression=True)
    book["Model_Directory"] = model_dir

    # load feature
    x = Data()
    x_fpaths = [d_dir / STR.fname_channels, d_dir / STR.fname_injury_crit]
    x.set_from_files(
        file_paths=x_fpaths,
        percentiles=[ft_perc],
        columns=USED_COLUMNS_AI_IN,
        idxs=sel_ids,
    )
    book["Feature_Files"] = x_fpaths

    # load target
    y_true: Data = Data()
    y_true_fpaths = [d_dir / STR.fname_injury_crit]
    y_true.set_from_files(
        file_paths=y_true_fpaths,
        percentiles=[tgt_perc],
        columns=USED_COLUMNS_AI_OUT,
        idxs=sel_ids,
    )
    book["Target_Files"] = y_true_fpaths
    y_true: pd.DataFrame = y_true.get_tabular()

    # predict
    y_pred: Data = cnn.predict(x=x)
    y_pred: pd.DataFrame = y_pred.get_tabular()
    y_pred.columns = USED_COLUMNS_AI_OUT

    # evaluate
    m = Metrics(fold_id=-1, mode="Test")
    result = m.r2_score(y_true=y_true, y_pred=y_pred)

    # store
    result.to_csv(STR.fname_results_csv)
    y_pred.to_parquet("y_pred_test.parquet")
    y_true.to_parquet("y_true_test.parquet")
    json_util.dump(obj=book, f_path=Path(STR.fname_results_info))

    # back to original directory
    os.chdir(cwd)


def cmd_parser() -> Tuple[Optional[Path], List[int]]:
    parser = argparse.ArgumentParser(description="Run test set experiment")
    parser.add_argument(
        "--sobol_dir",
        type=Path,
        default="",
        help="Provide for pure prediction mode - last field after _ must be target percentile",
    )
    parser.add_argument(
        "--perc",
        type=int,
        default=0,
        help="Switch to single target percentile mode (5 and 95 as target by default)",
    )
    args = parser.parse_args()
    return args.sobol_dir, [args.perc] if args.perc != 0 else [5, 95]


def test() -> None:
    s_dir, t_percs = cmd_parser()
    if s_dir == Path(""):
        for t_perc in t_percs:
            # reference metamodel
            _ = run(
                feature_percentiles=50,
                target_percentiles=t_perc,
                tgt="injury_criteria",
                database="doe_sobol_test_20240829_135200",
                ai_in=["channels"],
            )

            # main metamodel
            sobol_dir = run(
                feature_percentiles=50,
                target_percentiles=t_perc,
                tgt="injury_criteria",
                database="doe_sobol_20240705_194200",
                ai_in=["channels"],
            )

            # prediction (as subprocess for stability)

            LOG.info("Start prediction from model %s", sobol_dir)
            with mp.Pool(1) as pool:
                pool.starmap(do_prediction, [(sobol_dir.absolute(), t_perc, "doe_sobol_test_20240829_135200")])
            LOG.info("Prediction done")

    else:
        # prediction (as subprocess for stability)
        LOG.info("Start prediction")
        with mp.Pool(1) as pool:
            pool.starmap(do_prediction, [(s_dir, int(s_dir.stem.split("_")[-1]), "doe_sobol_test_20240829_135200")])
        LOG.info("Prediction done")


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
