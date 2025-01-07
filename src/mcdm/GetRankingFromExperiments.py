import logging
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Union, Optional, Tuple
import argparse

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
from src._StandardNames import StandardNames
from src.utils.custom_log import init_logger
from src.utils.PathChecker import PathChecker
import src.utils.json_util as json_util
from src.mcdm.Promethee import PrometheeSortingBased
from src.mcdm.ReadAlternatives import ReadAlternative

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class GetRankingFromExperiments:
    def __init__(self, b_path: Path, search_pattern: Optional[str] = None, files: Optional[List[str]] = None) -> None:
        self.b_path: Path = PathChecker().check_directory(path=b_path)

        if search_pattern is None:
            self.use: List[str] = files
        else:
            self.use: str = search_pattern

        self.data: Optional[pd.DataFrame] = None

    def get_data(self) -> None:
        LOG.info("Read data from %s with search pattern %s", self.b_path, self.use)
        reader = ReadAlternative(b_path=self.b_path, search_pattern=self.use)
        self.data = reader.get_data()

        LOG.info("Got data with  %s alternatives with %s criteria", *self.data.shape)

    def get_ranking(self) -> None:
        if self.data is None:
            LOG.critical("No data available - EXIT")
            sys.exit()
        else:
            LOG.info("Start ranking for %s alternatives with %s criteria", *self.data.shape)

        prom = PrometheeSortingBased()
        prom.get_data(in_info=self.data)
        prom.execute()

        ranking = pd.DataFrame(prom.ranking)
        ranking.index.name = STR.alternatives

        return pd.DataFrame(prom.ranking)


def eval_cmd_line() -> Tuple[Path, str]:
    parser = argparse.ArgumentParser(description="Get ranking from experiments")
    parser.add_argument("--b_path", "-p", type=Path, help="Base path to experiments", required=True)
    parser.add_argument("--search_pattern", "-s", type=str, help="Search pattern for results", required=True)
    parser.add_argument("--log_lvl", type=str, default=logging.INFO, help="Log level default: %(default)s")
    args = parser.parse_args()

    init_logger(log_lvl=args.log_lvl)

    return args.b_path, args.search_pattern


if __name__ == "__main__":
    r = GetRankingFromExperiments(*eval_cmd_line())
    r.get_data()
    LOG.info("Got Ranking:\n%s", r.get_ranking())
