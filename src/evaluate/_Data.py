import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import interp1d

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class Data:
    def __init__(self):
        self._temporal: Optional[pd.DataFrame] = None
        self._tabular: Optional[pd.DataFrame] = None
        self.n_samples: Optional[int] = None

        self._temporal_idx = (STR.time, STR.id)
        self._tabular_idx = (STR.id,)

    def get_temporal(self) -> Optional[pd.DataFrame]:
        return self._temporal

    def get_tabular(self) -> Optional[pd.DataFrame]:
        return self._tabular

    def get_temporal_3d(self, new_n_tsps: Optional[int] = None) -> Optional[Tuple[np.ndarray, int]]:
        if self._temporal is None:
            return None
        else:
            db = self.get_temporal_resampled(new_n_tsps=new_n_tsps)
            n_tsps_new_real = db.index.get_level_values(STR.time).nunique()

            # reshape
            db = np.array(np.split(db.values, n_tsps_new_real, axis=0)).transpose(1, 0, 2)
            LOG.debug("Reshape temporal data from %s to %s:\n%s", self._temporal.shape, db.shape, db)

            return db, n_tsps_new_real

    def get_temporal_resampled(self, new_n_tsps: Optional[int] = None) -> Optional[pd.DataFrame]:
        if self._temporal is None:
            return None
        else:
            db = self._temporal.unstack(level=STR.id)
            n_tsps = db.shape[0]

            if new_n_tsps is None or n_tsps == new_n_tsps:
                db = self._temporal
            elif new_n_tsps < n_tsps:
                LOG.info("Downsample temporal data from %s to %s", n_tsps, new_n_tsps)
                n_old = n_tsps
                step = int(np.round(n_old / new_n_tsps))
                n_old_per_new = int(step * new_n_tsps)

                if n_old < n_old_per_new:
                    dt = np.mean(np.diff(db.index))
                    db = pd.concat(
                        [db, pd.DataFrame(index=[db.index.max() + dt * (i + 1) for i in range(0, n_old_per_new - db.shape[0])])]
                    ).ffill()
                    db.columns.names = ["", STR.id]
                    db.index.name = STR.time
                elif n_old > n_old_per_new:
                    db = db.iloc[:n_old_per_new]

                db = (
                    db.rolling(window=step, step=step, center=True, min_periods=0)
                    .median()
                    .stack(level=STR.id, future_stack=True)
                    .reorder_levels(self._temporal.index.names)
                    .sort_index()
                )
            else:
                LOG.info("Upsample temporal data from %s to %s", n_tsps, new_n_tsps)
                tmp = self._temporal.unstack(level=STR.id)
                new_tmsps = np.linspace(tmp.index.min(), tmp.index.max(), new_n_tsps)
                db = (
                    pd.DataFrame(
                        interp1d(x=tmp.index, y=tmp.values, axis=0)(new_tmsps),
                        columns=tmp.columns,
                        index=pd.Index(new_tmsps, name=STR.time),
                    )
                    .stack(level=STR.id, future_stack=True)
                    .reorder_levels(self._temporal.index.names)
                    .sort_index()
                )
                del tmp

            return db

    def set_temporal_3d(self, data: np.ndarray, time_stmps: pd.Index, idx: pd.Index, columns: pd.Index):
        """Set temporal data from array

        Args:
            data (np.ndarray): data of shape (n_samples, n_tsps, n_channels)
            time_stmps (pd.Index): original time steps from training (n_tsps)
            idx (pd.Index): sample indices (n_samples)
            columns (pd.Index): Channel names (n_channels)
        """
        LOG.debug("Set temporal data with shape %s and time stamps %s", data.shape, time_stmps)

        if data.shape[1] != len(time_stmps):
            inner_tsps = np.linspace(time_stmps[0], time_stmps[-1], data.shape[1])
        else:
            inner_tsps = time_stmps

        idx_multi = pd.MultiIndex.from_product([idx, inner_tsps], names=[STR.id, STR.time])
        self.set_temporal(
            pd.DataFrame(
                data.reshape(data.shape[0] * data.shape[1], data.shape[2] if len(data.shape) == 3 else 1),
                index=idx_multi,
                columns=columns,
            )
        )

        if data.shape[1] != len(time_stmps):
            LOG.info("Resample temporal data from %s to %s", data.shape[1], len(time_stmps))
            self.set_temporal(db=self.get_temporal_resampled(new_n_tsps=len(time_stmps)))



    def reset_and_fill(self, tabular: Optional[pd.DataFrame] = None, temporal: Optional[pd.DataFrame] = None):
        self.set_tabular(tabular)
        self.set_temporal(temporal)

    def set_tabular(self, db: Optional[Union[pd.DataFrame, np.ndarray]]):
        if db is None:
            self._tabular = None
        elif isinstance(db, np.ndarray):
            self._tabular.loc[:, :] = db
        else:
            if db.index.names == self._tabular_idx:
                self._tabular = db[sorted(db.columns)].sort_index()
            else:
                LOG.error("Tabular data must have index with name %s", STR.id)
                LOG.debug("Tabular data:\n%s", db.head(2))
                self._tabular = None
        self.n_samples = self._tabular.shape[0] if self._tabular is not None else None
        self.log_status(show_temporal=False)

    def set_temporal(self, db: Optional[Union[pd.DataFrame, np.ndarray]]):
        if db is None:
            self._temporal = None
        elif isinstance(db, np.ndarray):
            self._temporal.loc[(slice(None), slice(None)), :] = db
        else:
            if set(db.index.names) == set(self._temporal_idx):
                self._temporal = db[sorted(db.columns)].reorder_levels(order=self._temporal_idx)
                self._temporal.sort_index(level=self._temporal_idx, inplace=True)
            else:
                LOG.error("Temporal data must have index with %s and %s", STR.time, STR.id)
                LOG.debug("Temporal data:\n%s", db.head(2))
        self.n_samples = self._temporal.shape[0] if self._temporal is not None else None

        self.log_status(show_tabular=False)

    def set_from_files(
        self,
        file_paths: List[Path],
        percentiles: Optional[List[int]],
        idxs: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ):
        """Read and concat data from files

        Args:
            file_paths (List[Path]): files paths to read
            percentiles (Optional[List[int]]): percentiles to select, None if DOE factors
            idxs (np.ndarray): ids to select)
        """
        # get data
        db_tabular, db_temporal = [], []
        for f_path in file_paths:
            available_columns = set(pq.read_schema(f_path).names) - {STR.id, STR.perc, STR.time}
            sought_columns = available_columns if columns is None else set(columns) & available_columns
            if sought_columns:
                db = self._read_db(file_path=f_path, percentiles=percentiles, idxs=idxs, columns=list(sought_columns))
                if len(db.index.names) == 1:
                    db_tabular.append(db)
                else:
                    db_temporal.append(db)
            else:
                LOG.warning("No matching columns to read from %s - SKIP", f_path)

        # concat data
        temporal = self._concat_data(dbs=db_temporal, file_paths=file_paths) if db_temporal else None
        tabular = self._concat_data(dbs=db_tabular, file_paths=file_paths) if db_tabular else None

        # set
        self.set_tabular(tabular)
        self.set_temporal(temporal)

    def log_status(self, show_temporal: bool = True, show_tabular: bool = True, ignore_log_lvl: bool = False) -> None:
        LOG.info(
            "Got data temporal %s and tabular %s",
            self._temporal.shape if isinstance(self._temporal, pd.DataFrame) else None,
            self._tabular.shape if isinstance(self._tabular, pd.DataFrame) else None,
        )
        if show_tabular:
            LOG.debug(
                "Current data container tabular %s :\n%s", None if self._tabular is None else self._tabular.shape, self._tabular
            )
            if ignore_log_lvl:
                LOG.info(
                    "Current data container tabular %s :\n%s",
                    None if self._tabular is None else self._tabular.shape,
                    self._tabular,
                )
        if show_temporal:
            LOG.debug(
                "Current data container temporal %s:\n%s",
                None if self._temporal is None else self._temporal.shape,
                self._temporal,
            )
            if ignore_log_lvl:
                LOG.info(
                    "Current data container temporal %s:\n%s",
                    None if self._temporal is None else self._temporal.shape,
                    self._temporal,
                )

    def _concat_data(self, dbs: List[pd.DataFrame], file_paths: List[Path]) -> pd.DataFrame:
        """Concat data

        Args:
            dbs (List[pd.DataFrame]): data frames to concat
            file_paths (List[Path]): file paths as reference

        Returns:
            pd.DataFrame: concatenated data, index with ID, optional multi index with TIME
        """
        if len(dbs) == 1:
            dbs = dbs[0]
        else:
            for i, db in enumerate(dbs):
                db.rename(columns={col: f"{col}__{file_paths[i].stem}" for col in db.columns}, inplace=True)
            dbs = pd.concat(dbs, axis=1)

        return dbs

    def _read_db(
        self,
        file_path: Path,
        percentiles: Optional[List[int]],
        idxs: Optional[np.ndarray],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read single file, ravel if multiple percentiles

        Args:
            file_path (Path): path of file
            percentiles (Optional[List[int]]): percentiles to select, None if DOE factors
            idxs (np.ndarray): ids to select
            columns (Optional[List[str]]): columns to load. Defaults to None

        Returns:
            pd.DataFrame: read data, index with ID, optional multi index with TIME
        """
        filters = []
        if idxs is not None:
            filters.append((STR.id, "in", idxs))
        if percentiles is not None:
            filters.append((STR.perc, "in", percentiles))

        LOG.info("Read data from %s", file_path)
        if percentiles is None:
            # doe
            # expect ID as index and factors as columns
            db = pd.read_parquet(path=file_path, filters=filters, columns=columns).apply(pd.to_numeric, downcast="float")
        else:
            # simulation data
            # Tabular case: expect ID and PERC as index and features as columns
            # Temporal case: expect ID, PERC and Time as index and channels as columns
            db = pd.read_parquet(
                path=file_path,
                filters=filters,
                columns=columns,
            ).apply(pd.to_numeric, downcast="float")

            if len(percentiles) == 1:
                # single percentile
                db.index = db.index.droplevel(level=STR.perc)
            else:
                # multiple percentiles
                db = db.unstack(level=STR.perc)
                db.columns = db.columns.to_flat_index()

        LOG.info("Got data of shape %s and size %.3fMB", db.shape, db.memory_usage().sum() / 1024**2)

        return db

    def append(self, other) -> None:
        if isinstance(other.get_tabular(), pd.DataFrame):
            if self.get_tabular() is None:
                self.set_tabular(other.get_tabular())
            else:
                self.set_tabular(pd.concat([self.get_tabular(), other.get_tabular()], axis=0))
        else:
            if self.get_temporal() is None:
                self.set_temporal(other.get_temporal())
            else:
                self.set_temporal(pd.concat([self.get_temporal(), other.get_temporal()], axis=0))
