import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
from src._StandardNames import StandardNames
from src.evaluate.Results import Results
from src.mcdm.Criteria import Criteria

LOG: logging.Logger = logging.getLogger(__name__)


class ResultsGlobal:
    def __init__(
        self,
        n_crash_simulations: int,
        n_occupant_simulations: int,
        n_target_anthros: int,
        frac_train=float,
        frac_test=float,
    ):
        LOG.debug("Init Global Results")
        self.__str = StandardNames()
        self.__scores: Optional[pd.DataFrame] = None
        self.data_info: Dict[int, Any] = {}
        self.data_split: Dict[int, Any] = {}
        self.mcdm_criteria: Criteria = Criteria(
            n_crash_simulations=n_crash_simulations,
            n_occupant_simulations=n_occupant_simulations,
            n_target_anthros=n_target_anthros,
            frac_train=frac_train,
            frac_test=frac_test,
        )

        # environment
        self.python: str = sys.version
        git_hash: str = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC_DIR).decode("ascii")
        branch: str = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=SRC_DIR).decode("ascii")
        self.git: Dict[str, str] = {
            self.__str.hash: git_hash.strip(),
            "Branch": branch.strip(),
        }

    def update_scores(self, scores: pd.DataFrame) -> None:
        if self.__scores is None:
            self.__scores = scores
        else:
            self.__scores = pd.concat([self.__scores, scores])

    def update_local_info(self, results: Results) -> None:
        self.update_scores(results.get_scores())
        self.data_info[results.fold_id] = results.get_metadata()

    def get_scores(self) -> pd.DataFrame:
        self.data_info[self.__str.metrics] = self.__scores.groupby(self.__str.data).median().to_dict()
        return self.__scores

    def add_data_split_info(self, split_type: str, split_paras: dict, fold_info: Dict[int, Dict[str, int]]):
        self.data_split["split_type"] = split_type
        self.data_split["split_paras"] = split_paras
        self.data_split["fold_info"] = fold_info
