import logging
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd

LOG: logging.Logger = logging.getLogger(__name__)


class Results:
    def __init__(self, fold_id: int):
        LOG.debug("Init Results")
        self.__scores: Optional[pd.DataFrame] = None
        self.fold_id: int = fold_id
        self.comp_time_fit: Optional[float] = None
        self.comp_time_pred_train: Optional[float] = None
        self.comp_time_pred_test: Optional[float] = None
        self.hyperparameter: Optional[Any] = None
        self.y_true_train_fpath: Optional[Path] = None
        self.y_pred_train_fpath: Optional[Path] = None
        self.y_true_test_fpath: Optional[Path] = None
        self.y_pred_test_fpath: Optional[Path] = None

    def update_scores(self, scores: pd.DataFrame) -> None:
        if self.__scores is None:
            self.__scores = scores
        else:
            self.__scores = pd.concat([self.__scores, scores])

    def get_scores(self) -> pd.DataFrame:
        return self.__scores

    def get_metadata(self) -> Dict[str, Any]:
        return {key: val for key, val in self.__dict__.items() if not key.startswith("_")}
