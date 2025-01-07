import argparse
import datetime
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src.utils.Csv import Csv
from src._StandardNames import StandardNames
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def get_directory() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, type=Path)
    args = parser.parse_args()

    return PathChecker().check_directory(path=args.directory)


def read_data(b_path: Path) -> None:
    data = []
    LOG.info("Get files from %s", b_path)
    files = sorted(b_path.rglob("extracted.csv.zip"))
    f_hashes = {}

    LOG.info("Read %s files", len(files))
    for file in tqdm(files):
        db: pd.DataFrame = Csv(csv_path=file, compress=True).read().apply(pd.to_numeric, downcast="float")

        d_name = file.parent.stem
        db[STR.id] = int(d_name.split("_")[2][6:])  # e.g. VH_AF05_Config0001_THI_RESULT to 1
        perc = int(d_name.split("_")[1][2:])
        db[STR.perc] = perc
        db.set_index([STR.id, STR.perc], append=True, inplace=True)

        db.rename(columns={col: col.replace(f"VI{perc:02d}", "OCCU") for col in db.columns}, inplace=True)
        db.drop(columns=[col for col in db.columns if "OCCU" not in col or not col.endswith("D")], inplace=True)

        f_hashes[d_name[3:-11]] = {STR.path: file, STR.hash: hash_file(fpath=file)}

        data.append(db)

    LOG.info("Read %s files - Concat", len(data))
    data = pd.concat(data, ignore_index=False, axis=0, copy=False)
    data.sort_index(inplace=True)
    LOG.info("Read files - Done - got %s\n%s", data.shape, data)

    out_file = b_path / STR.fname_channels
    LOG.info("Write to %s", out_file)
    data.to_parquet(out_file, index=True)

    # bookkeeping
    book_file = b_path / STR.fname_results_info
    book = {
        STR.creation: str(datetime.datetime.now()),
        STR.output: {
            STR.channelss: {STR.path: out_file, STR.hash: hash_file(fpath=out_file)},
        },
        STR.input: f_hashes,
    }
    LOG.info("Write bookkeeping to %s", book_file)
    json_util.dump(obj=book, f_path=book_file)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    LOG.info("Start")
    read_data(b_path=get_directory())
    LOG.info("Done")
