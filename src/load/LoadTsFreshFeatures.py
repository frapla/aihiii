import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tsfresh import extract_features

import numpy as np
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
import src.utils.hash_file as hash_file
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.utils.PathChecker import PathChecker
from src.utils.ParquetHandler import ParquetHandler

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def main():
    # cmd line
    directory, percentile = eval_cmd_line()

    # paths
    directory = PathChecker().check_directory(directory)
    channel_fpath = PathChecker().check_file(directory / STR.fname_channels)
    info_fpath = PathChecker().check_file(directory / STR.fname_results_info)

    # check hash
    hash_is = hash_file.hash_file(channel_fpath)
    hash_should = json_util.load(info_fpath)[STR.output][STR.channelss][STR.hash]
    same_hash = hash_is == hash_should
    if same_hash:
        LOG.info("Hashes of channel file is the same")
    else:
        raise ValueError("Hashes of channel file is different")

    # relevant columns
    relevant_columns = [
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
        "03FACE0000OCCUFORD",
        "03SHLDRIUPOCCUFORD",
        "03SHLDLEUPOCCUFORD",
        "03SHLDRILOOCCUFORD",
        "03SHLDLELOOCCUFORD",
        "03CHSTRIUPOCCUFORD",
        "03CHSTLEUPOCCUFORD",
        "03CHSTRILOOCCUFORD",
        "03CHSTLELOOCCUFORD",
        "03KNEERI00OCCUFORD",
        "03KNEELE00OCCUFORD",
    ]
    LOG.info("Relevant columns %s", len(relevant_columns))

    # load data
    LOG.info("Loading channels")
    db = ParquetHandler(path=channel_fpath).read(percentiles=[percentile], columns=relevant_columns).reset_index()
    LOG.info("Loaded channels %s", db.shape)

    # extract features
    LOG.info("Extracting features")
    extracted_features: pd.DataFrame = extract_features(db, column_id=STR.id, column_sort=STR.time, disable_progressbar=False)
    del db
    extracted_features.index.name = STR.id
    LOG.info("Extracted features %s", extracted_features.shape)

    # remove NaNs and columns with std < 1e-6
    LOG.info("Drop Useless Columns")
    extracted_features.dropna(axis=1, inplace=True)
    standard_devs = extracted_features.std()
    extracted_features.drop(columns=list(standard_devs[standard_devs.le(1e-6)].index), inplace=True)
    LOG.info("Dropped columns %s", extracted_features.shape)

    # store
    f_path = directory / f"tsfresh_features_{percentile}.parquet"
    LOG.info("Store to %s", f_path)
    extracted_features.to_parquet(f_path, index=True)
    # bookkeeping
    book = json_util.load(info_fpath)
    book[STR.creation] = str(datetime.datetime.now())
    book[STR.output][f"TSFRESH Features {percentile}th"] = {STR.path: f_path, STR.hash: hash_file.hash_file(f_path)}

    LOG.info("Store info to %s", info_fpath)
    json_util.dump(obj=book, f_path=info_fpath)

    LOG.info("Done")


def eval_cmd_line() -> Tuple[Path, Path]:
    """Evaluate and check command line arguments

    Raises:
        FileNotFoundError: one of the directories does not exist

    Returns:
        Tuple[Path, Path]: raw data directory, processed data directory
    """
    # init
    parser = argparse.ArgumentParser(description="Load data for DOE2FE")

    # arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Directory injury data to fetch data from",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--percentile",
        type=int,
        help="Dummy Percentile (default: %(default)s)",
        required=False,
        default=50,
    )
    parser.add_argument(
        "--log_lvl",
        default=logging.INFO,
        help="Log level (default: %(default)s)",
        required=False,
        type=int,
    )

    # parse
    args = parser.parse_args()

    # set log level
    custom_log.init_logger(log_lvl=args.log_lvl)

    return args.directory, args.percentile


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    main()
