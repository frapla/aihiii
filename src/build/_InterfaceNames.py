import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
import logging

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.evaluate._Data import Data
import src.utils.json_util as json_util
from src._StandardNames import StandardNames

STR: StandardNames = StandardNames()
LOG: logging.Logger = logging.getLogger(__name__)


class InterfaceNames:
    def __init__(self) -> None:
        self.temporal_features: Optional[List[str]] = None
        self.tabular_features: Optional[List[str]] = None
        self.temporal_target: Optional[List[str]] = None
        self.tabular_target: Optional[List[str]] = None
        self.orig_target_tsp: Optional[int] = None

    def set_features(self, data: Data) -> None:
        self._set_from_data(data, is_feature=True)

    def set_target(self, data: Data) -> None:
        self._set_from_data(data, is_feature=False)

    def compare_features(self, data: Data) -> bool:
        return self._compare_with_data(data, is_feature=True)

    def compare_target(self, data: Data) -> bool:
        return self._compare_with_data(data, is_feature=False)

    def get_params(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}

    def store_params(self) -> None:
        LOG.info("Store interface parameters in %s", STR.fname_interface)
        json_util.dump(obj=self.get_params(), f_path=Path(STR.fname_interface))

    def load_params(self, inter_dir: Optional[Path] = None) -> None:
        if inter_dir is None:
            f_path = Path(STR.fname_interface)
        else:
            f_path = inter_dir / STR.fname_interface
        LOG.info("Load interface parameters from %s", f_path)
        self.__dict__.update(json_util.load(f_path))

    def _set_from_data(self, data: Data, is_feature: bool) -> None:
        db = data.get_tabular()
        if isinstance(db, pd.DataFrame):
            if is_feature:
                self.tabular_features = db.columns.tolist()
            else:
                self.tabular_target = db.columns.tolist()

        db = data.get_temporal()
        if isinstance(db, pd.DataFrame):
            if is_feature:
                self.temporal_features = db.columns.tolist()
            else:
                self.temporal_target = db.columns.tolist()

    def _compare_with_data(self, data: Data, is_feature: bool) -> bool:
        check_tabular, check_temporal = True, True

        db = data.get_tabular()
        if isinstance(db, pd.DataFrame):
            if is_feature:
                check_tabular = self.tabular_features == db.columns.tolist()
            else:
                check_tabular = self.tabular_target == db.columns.tolist()

        db = data.get_temporal()
        if isinstance(db, pd.DataFrame):
            if is_feature:
                check_temporal = self.temporal_features == db.columns.tolist()
            else:
                check_temporal = self.temporal_target == db.columns.tolist()

        return check_tabular and check_temporal
