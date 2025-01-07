import logging
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
from src._StandardNames import StandardNames
from src.utils.custom_log import init_logger
from src.utils.PathChecker import PathChecker
import src.utils.json_util as json_util

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class ReadAlternative:
    def __init__(self, b_path: Path, search_pattern: Union[str, List[str]]):
        self.b_path: Path = PathChecker().check_directory(path=b_path)
        self.search_pattern: Union[str, List[str]] = search_pattern

    def get_data(self) -> pd.DataFrame:
        if isinstance(self.search_pattern, list):
            result_dirs = [self.b_path / p for p in self.search_pattern]
        else:
            result_dirs = self.b_path.glob(self.search_pattern)

        results = {}
        for result_dir in result_dirs:
            res_path = result_dir / STR.fname_results
            if res_path.is_file():
                LOG.info("Reading %s", res_path)
                content = json_util.load(f_path=res_path)
                results[result_dir.stem] = content[STR.result][STR.mcdm]
            else:
                LOG.warning("No results found in %s", result_dir)

        results = pd.DataFrame(results)
        results.index.name = STR.criteria
        results.columns.name = STR.alternatives

        return pd.DataFrame(results)


if __name__ == "__main__":
    # test
    init_logger(log_lvl=logging.DEBUG)
    reader = ReadAlternative(b_path=Path("experiments"), search_pattern="*_lstm_treesearch_*")

    print(reader.get_data())
