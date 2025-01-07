import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from src.utils.injury_limits import get_limits_euro_ncap
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()

CLASS_THRESHOLDS: Dict[int, Dict[int, int]] = {
    5: {
        2: [4],
        3: [4, 4.3],
        5: [1, 4, 4.2, 4.4],
        7: [1, 2, 3, 4, 4.2, 4.4],
    },
    95: {
        2: [1],
        3: [0.8, 4],
        5: [0.8, 1, 4, 4.3],
        7: [0.8, 1, 2, 3, 4, 4.3],
    },
}

RENAMER: Dict[int, Dict[int, Dict[int, str]]] = {
    5: {
        2: {
            0: "0: P<4",
            1: "1: P>=4",
        },
        3: {
            0: "0: P<4",
            1: "1: 4>=P<4.3",
            2: "2: P>=4.3",
        },
        5: {
            0: "0: P<1",
            1: "1: 1>=P<4",
            2: "2: 4>=P<4.2",
            3: "3: 4.2>=P<4.4",
            4: "4: P>=4.4",
        },
        7: {
            0: "0: P<1",
            1: "1: 1>=P<2",
            2: "2: 2>=P<3",
            3: "3: 3>P<4",
            4: "4: 4>=P<4.2",
            5: "5: 4.2>=P<4.4",
            6: "6: P>=4.4",
        },
    },
    95: {
        2: {
            0: "0: P<1",
            1: "1: P>=1",
        },
        3: {
            0: "0: P<0.8",
            1: "1: 0.8>=P<4",
            2: "2: P>=4",
        },
        5: {
            0: "0: P<0.8",
            1: "1: 0.8>=P<1",
            2: "2: 1>=P<4",
            3: "3: 4>=P<4.3",
            4: "4: P>=4.3",
        },
        7: {
            0: "0: P<0.8",
            1: "1: 0.8>=P<1",
            2: "2: 1>=P<2",
            3: "3: 2>=P<3",
            4: "4: 3>=P<4",
            5: "5: 4>=P<4.3",
            6: "6: V>=4.3",
        },
    },
}

NCAP_BUFFERS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


class LoadForClassification:
    def __init__(self, data_directory: Path):
        self.data_directory: Path = PathChecker().check_directory(data_directory)
        self.injury_fpath: Path = PathChecker().check_file(self.data_directory / STR.fname_injury_crit)
        self.info_fpath: Path = PathChecker().check_file(self.data_directory / STR.fname_results_info)

        self.check_hash()

        self.euro_ncap_thresholds: pd.DataFrame = get_limits_euro_ncap(buffers=NCAP_BUFFERS)

    def check_hash(self) -> None:
        hash_is = hash_file.hash_file(self.injury_fpath)
        hash_should = json_util.load(self.info_fpath)[STR.output][STR.injury_criteria][STR.hash]

        same_hash = hash_is == hash_should
        if same_hash:
            LOG.info("Hashes of injury files are the same")
        else:
            raise ValueError("Hashes of injury files are different")

    def injury_value_to_points(self, percentile: int) -> pd.DataFrame:
        pnts = sorted(self.euro_ncap_thresholds.index.get_level_values("POINTS").unique())

        classes = []
        for inj_crit in self.euro_ncap_thresholds.columns:
            limits = [(self.euro_ncap_thresholds.loc[(pnts[0], percentile), inj_crit], np.inf)]
            limits.extend(
                [
                    (
                        self.euro_ncap_thresholds.loc[(pnts[i + 1], percentile), inj_crit],
                        self.euro_ncap_thresholds.loc[(pnts[i], percentile), inj_crit],
                    )
                    for i in range(len(pnts) - 1)
                ]
            )
            limits.append([-np.inf, self.euro_ncap_thresholds.loc[(pnts[-1], percentile), inj_crit]])

            db = pd.read_parquet(self.injury_fpath, columns=[inj_crit], filters=[(STR.perc, "==", percentile)])

            collector = db[inj_crit].ge(np.inf)
            for pnt, (lim_lo, lim_up) in zip([0] + pnts, limits):
                collector += db[inj_crit].between(lim_lo, lim_up) * pnt
            classes.append(pd.DataFrame(collector))
        return pd.concat(classes, axis=1)

    def points_to_classes(self, n_classes: int, euro_points: pd.DataFrame, percentile: int):
        thres = CLASS_THRESHOLDS[percentile][n_classes]
        result = np.zeros_like(euro_points)
        for th in thres:
            result += euro_points.ge(th).astype(int)
        result = pd.DataFrame(result, index=euro_points.index, columns=euro_points.columns)

        return result

    def make_classes(self):
        LOG.info("Create classes")
        f_paths = {}
        for n_classes in CLASS_THRESHOLDS[5].keys():
            LOG.info("Create classes for %s classes", n_classes)
            db_percentiles = []
            for percentile in CLASS_THRESHOLDS.keys():
                LOG.info("Create classes for %s percentile", percentile)
                euro_points = self.injury_value_to_points(percentile=percentile)
                db_percentiles.append(self.points_to_classes(n_classes=n_classes, euro_points=euro_points, percentile=percentile))
            db_percentiles = pd.concat(db_percentiles, axis=0)

            # store
            f_paths[n_classes] = self.data_directory / f"injury_criteria_classes_{n_classes}.parquet"
            LOG.info("Store classes to %s", f_paths[n_classes])
            db_percentiles.to_parquet(path=f_paths[n_classes], index=True)

        LOG.info("Update bookkeeper")
        book = json_util.load(self.info_fpath)
        book[STR.creation] = str(datetime.datetime.now())
        for n_classes, f_path in f_paths.items():
            book[STR.output][f"Injury Criteria Classes {n_classes}"] = {STR.path: f_path, STR.hash: hash_file.hash_file(f_path)}

        LOG.info("Store info to %s", self.info_fpath)
        json_util.dump(obj=book, f_path=self.info_fpath)


def main():
    """run main function"""
    directory = eval_cmd_line()
    LOG.info("Load data")

    loader = LoadForClassification(data_directory=directory)
    loader.make_classes()

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

    return args.directory


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    main()
