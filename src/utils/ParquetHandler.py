import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).absolute().parent))
from src._StandardNames import StandardNames
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


class ParquetHandler:
    def __init__(self, path: Path):
        self.path: Path = PathChecker().check_file(path)

        self._str: StandardNames = StandardNames()

    def get_columns(self) -> List[str]:
        LOG.debug("Reading columns from %s", self.path)
        column_names = pq.read_schema(self.path).names
        LOG.debug("Got %s columns from %s", len(column_names), self.path)

        return column_names

    def read(
        self,
        ids: Optional[List[int]] = None,
        percentiles: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        columns: Optional[List[str]] = None,
        downcast: bool = True,
    ) -> pd.DataFrame:
        LOG.debug("Reading data from %s", self.path)

        if columns is None:
            used_cols = None
        else:
            used_cols = list(set(self.get_columns()) & set(columns))
            if len(used_cols) < len(columns):
                LOG.warning("%s columns are not %s", len(columns) - len(used_cols), set(columns) - set(used_cols))

        filters = []
        if ids is not None:
            filters.append((self._str.id, "in", ids))
        if percentiles is not None:
            filters.append((self._str.perc, "in", percentiles))
        if time_range is not None:
            filters.append([(self._str.time, ">=", time_range[0]), (self._str.time, "<=", time_range[1])])
        if len(filters) == 0:
            filters = None

        db = pd.read_parquet(path=self.path, filters=filters, columns=used_cols)
        if downcast:
            db = db.apply(pd.to_numeric, downcast="float")

        if percentiles is not None and len(percentiles) == 1:
            db = db.droplevel(self._str.perc)
        if ids is not None and len(ids) == 1:
            db = db.droplevel(self._str.id)
        if time_range is not None and time_range[0] == time_range[1]:
            db = db.droplevel(self._str.time)

        LOG.debug("Got shape %s from %s", db.shape, self.path)
        return db
