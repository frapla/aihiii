import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
import polars as pl

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

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class LoadForClassification:
    def __init__(self, data_directory: Path, injury_crit: str, threshold: float, percentile: Literal[5, 50, 95]):
        self.data_directory: Path = PathChecker().check_directory(data_directory)
        self.injury_fpath: Path = PathChecker().check_file(self.data_directory / STR.fname_injury_crit)
        self.info_fpath: Path = PathChecker().check_file(self.data_directory / STR.fname_results_info)
        self.injury_crit: str = injury_crit
        self.threshold: float = threshold
        self.percentile: int = percentile

        self.check_hash()

    def check_hash(self) -> None:
        hash_is = hash_file.hash_file(self.injury_fpath)
        hash_should = json_util.load(self.info_fpath)[STR.output][STR.injury_criteria][STR.hash]

        same_hash = hash_is == hash_should
        if same_hash:
            LOG.info("Hashes of injury files are the same")
        else:
            LOG.error("Hash of %s: %s", self.injury_fpath, hash_is)
            LOG.error("Hash of %s: %s", self.info_fpath, hash_should)
            raise ValueError("Hashes of injury files are different")

    def make_classes(self):
        # load data
        LOG.info("Load data from %s", self.injury_fpath)
        db = (
            (
                pl.scan_parquet(self.injury_fpath)
                .select([pl.col(self.injury_crit) > self.threshold, STR.perc, STR.id])
                .filter(pl.col(STR.perc) == self.percentile)
                .cast(pl.Int32)
                .collect()
            )
            .to_pandas()
            .set_index([STR.id, STR.perc])
        )
        LOG.info("Got %s with:\n%s", db.shape, db[self.injury_crit].value_counts())

        # store data
        f_path = (
            self.data_directory / f"injury_criteria_classes_{self.injury_crit}_perc{self.percentile:02d}_THRES_{self.threshold:.2f}_2".replace(".", "_")
        ).with_suffix(".parquet")
        LOG.info("Store data to %s", f_path)
        db.to_parquet(f_path, index=True)

        # bookkeeper
        LOG.info("Update bookkeeper")
        book = json_util.load(self.info_fpath)
        book[STR.creation] = str(datetime.datetime.now())
        book[STR.output][f_path.stem] = {STR.path: f_path, STR.hash: hash_file.hash_file(f_path)}
        LOG.info("Store info to %s", self.info_fpath)
        json_util.dump(obj=book, f_path=self.info_fpath)


def main():
    """run main function"""
    cmd = CmdLineArgs()
    LOG.info("Load data")

    loader = LoadForClassification(
        data_directory=cmd.directory, injury_crit=cmd.injury_crit, threshold=cmd.threshold, percentile=cmd.percentile
    )
    loader.make_classes()

    LOG.info("Done")


class CmdLineArgs:
    def __init__(self):
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
            "-i",
            "--injury_crit",
            type=str,
            help="Injury criterion name to use",
            required=True,
        )
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            help="Threshold to split the data",
            required=True,
        )
        parser.add_argument(
            "-p",
            "--percentile",
            type=int,
            help="Percentile to apply the threshold",
            required=True,
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
        self.directory: Path = PathChecker().check_directory(path=args.directory)
        self.injury_crit: str = args.injury_crit
        self.threshold: float = args.threshold
        self.percentile: int = args.percentile
        self.log_lvl: int = args.log_lvl

        # set log level
        custom_log.init_logger(log_lvl=self.log_lvl)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    main()
