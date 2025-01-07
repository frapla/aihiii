import argparse
import datetime
import logging
import multiprocessing as mp
import sys
import time
from itertools import product, starmap
from pathlib import Path
from typing import List, Tuple
import polars as pl

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.data.fe_processing.InjuryCalculator import InjuryCalculator
from src.data.fe_processing.IsoMme import IsoMme
from src.utils.Csv import Csv
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def get_directory() -> Tuple[Path, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, type=Path)
    parser.add_argument("-c", "--cpu", required=False, type=int, default=1)
    args = parser.parse_args()

    return PathChecker().check_directory(path=args.directory), args.cpu


def run_parallel(idx: int, perc: int, channels_file: Path) -> pd.DataFrame:
    LOG.info("Get perc %s and idx %s from %s", perc, idx, channels_file)
    q1 = (
        pl.scan_parquet(channels_file)
        .filter(pl.col(STR.perc) == perc)
        .filter(pl.col(STR.id) == idx)
        .select(pl.exclude(STR.perc, STR.id))
        .rename(lambda s: s.replace("OCCU", "H3" + f"{int(perc):02d}"))
    )
    db = q1.collect()

    LOG.debug("Got %s:\n%s", db.shape, db)

    inj = InjuryCalculator(data=db, mme=IsoMme(dummy_type="H3", dummy_percentile=int(perc), dummy_position="03"), cfc="D")
    inj.calculate()

    # format
    db = pd.DataFrame(inj.injury_crit, index=pd.MultiIndex.from_tuples([(idx, perc)], names=[STR.id, STR.perc]))

    LOG.info("Got injury_crit %s from perc %s and idx %s from %s", db.shape, perc, idx, channels_file)

    return db


def process_data(b_path: Path, n_cpu: int = 1) -> None:
    # book
    book_path = PathChecker().check_file(b_path / STR.fname_results_info)
    LOG.info("Read book from %s", book_path)
    book = json_util.load(book_path)

    # check file
    LOG.info("Check")
    channels_file = PathChecker().check_file(b_path / Path(book[STR.output][STR.channelss][STR.path]).name)
    channel_hash = hash_file(fpath=channels_file)
    channel_hash_ref = book[STR.output][STR.channelss][STR.hash]
    if channel_hash == channel_hash_ref:
        LOG.info("Channel file %s is up to date", channels_file)
    else:
        raise ValueError(f"Channel file {channels_file} is not up to date - {channel_hash} != {channel_hash_ref}")

    # get metadata
    LOG.info("Get metadata from %s", channels_file)
    db = pd.read_parquet(channels_file, columns=[STR.perc, STR.id]).droplevel(STR.time).index.unique()
    cases = [list(w) + [channels_file] for w in db.to_list()]
    del db

    if n_cpu != 1:
        n_cpu = mp.cpu_count() if (n_cpu == -1 or n_cpu > mp.cpu_count()) else n_cpu
        with mp.Pool(n_cpu, maxtasksperchild=1) as pool:
            injury_crit = pool.starmap(run_parallel, cases)
    else:
        injury_crit = list(starmap(run_parallel, cases))

    LOG.info("Got injury_crit %s - Concat", len(injury_crit))
    injury_crit = pd.concat(injury_crit, copy=False, axis=0, ignore_index=False)
    LOG.info("Got injury_crit %s:\n%s", injury_crit.shape, injury_crit)

    # store
    f_path_injury = b_path / STR.fname_injury_crit
    LOG.info("Write %s", f_path_injury)
    injury_crit.to_parquet(f_path_injury, index=True)
    inj_hash = hash_file(fpath=f_path_injury)

    # document
    LOG.info("Document results in %s", book_path)
    book[STR.creation] = str(datetime.datetime.now())
    book[STR.output] = {
        STR.channelss: {STR.path: channels_file, STR.hash: channel_hash},
        STR.injury_criteria: {STR.path: f_path_injury, STR.hash: inj_hash},
    }
    json_util.dump(f_path=book_path, obj=book)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    LOG.info("Start")
    process_data(*get_directory())
    LOG.info("Done")
