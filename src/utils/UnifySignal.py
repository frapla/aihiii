import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

sys.path.append(str(Path(__file__).absolute().parent))
import custom_log

LOG: logging.Logger = logging.getLogger(__name__)


class UnifySignal:
    def __init__(
        self,
        target_tend_ms: float,
        target_sampling_rate_ms: float,
        target_tstart_ms: float = 0,
        pad_add_length_fac: float = 0.1,
    ) -> None:
        """Interpolate and pad signals to a unified time axis

        Args:
            target_tend_ms (float): end time of unified data
            target_sampling_rate_ms (float): sampling rate of unified data
            target_tstart_ms (float, optional): start time of unified data. Defaults to 0.
            padwith (Union[float, str], optional): padding mode. Defaults to "median".
        """

        # unification targets
        self.target_tend_ms = target_tend_ms
        self.target_tstart_ms = target_tstart_ms
        self.target_sampling_rate_ms = target_sampling_rate_ms
        self.target_tsps = np.arange(
            target_tstart_ms,
            target_tend_ms + target_sampling_rate_ms,
            target_sampling_rate_ms,
        )

        # init
        self.is_short = False
        self.pad_add_length_fac = pad_add_length_fac

    def unify(self, db: pd.DataFrame) -> pd.DataFrame:
        """Pad and interpolate data

        Args:
            db (pd.DataFrame): signals of shape (n time stamps, n channels), index is time

        Returns:
            pd.DataFrame: interpolated signals of shape (n time stamps target, n channels), index is time
        """
        LOG.debug("Try to unify data")
        # ensure order
        db = db.sort_index()

        # pad signal
        db_pad = self.__pad(db=db)

        # interpolate
        interpolated = self.__interpolate(db=db_pad)

        LOG.debug("Unified data to shape %s", interpolated.shape)
        return interpolated

    def __pad(self, db: pd.DataFrame) -> pd.DataFrame:
        """Pad data by median to target end time

        Args:
            db (pd.DataFrame): signals of shape (n time stamps, n channels), index is time

        Returns:
            pd.DataFrame: padded signals of shape (n time stamps padded, n channels), index is time
        """
        LOG.debug("Pad data from shape %s", db.shape)
        # get padding parameters
        pad_add_length = self.pad_add_length_fac * (self.target_tend_ms - self.target_tstart_ms)
        target_t_min = self.target_tstart_ms - pad_add_length
        target_t_max = self.target_tend_ms + pad_add_length
        pad_stat_length = np.ceil(db.shape[0] * 0.01)
        LOG.debug(
            "Pad data from %s to %sms using median statistic with %s", db.shape, (target_t_min, target_t_max), pad_stat_length
        )

        # pad left parameters
        time_stps = db.index.to_numpy()
        if target_t_min < time_stps.min():
            time_pad_left = np.arange(
                start=target_t_min,
                stop=time_stps.min(),
                step=self.target_sampling_rate_ms,
            )
        else:
            time_pad_left = np.array([])

        # pad right parameters
        if target_t_max > time_stps.max():
            time_pad_right = np.arange(
                start=time_stps.max() + self.target_sampling_rate_ms,
                stop=target_t_max + self.target_sampling_rate_ms,
                step=self.target_sampling_rate_ms,
            )
        else:
            time_pad_right = np.array([])

        # pad time step
        pad_time_stps = np.concatenate((time_pad_left, time_stps, time_pad_right))
        LOG.debug("Pad time steps from %s to %s", time_stps.shape, pad_time_stps.shape)

        # pad data
        db_pad = np.pad(
            db.values,
            pad_width=((time_pad_left.shape[0], time_pad_right.shape[0]), (0, 0)),
            mode="median",
            stat_length=[pad_stat_length] * 2,
        )
        LOG.debug("Pad data from %s to %s", db.shape, db_pad.shape)

        # assemble back to pandas
        db_padded = pd.DataFrame(db_pad, index=pad_time_stps, columns=db.columns)

        LOG.debug("Padded shape is %s", db_padded.shape)
        return db_padded

    def __interpolate(self, db: pd.DataFrame) -> pd.DataFrame:
        """Interpolate on target time stamps

        Args:
            db (pd.DataFrame): signals of shape (n time stamps, n channels), index is time

        Returns:
            pd.DataFrame: interpolated signals of shape (n time stamps target, n channels), index is time
        """
        sr_rs = np.diff(db.index)
        LOG.debug("Interpolate mixed sampling rate (min=%sms, mean=%sms, max=%sms)", sr_rs.min(), sr_rs.mean(), sr_rs.max())
        # ensure order
        channel_names = sorted(db.columns)

        # interpolate
        interpolator = Akima1DInterpolator(x=db.index, y=db[channel_names].values, axis=0)
        interpolated = interpolator(self.target_tsps)

        # assemble back to pandas
        db_inter = pd.DataFrame(interpolated, index=self.target_tsps, columns=channel_names)  # .fillna(0)

        LOG.debug(
            "Interpolated on constant sampling rate %s to shape %s mean %s",
            self.target_sampling_rate_ms,
            db_inter.shape,
            np.mean(interpolated),
        )
        return db_inter


def test():
    db_raw = pd.DataFrame({"A": [1, 2, 3, 43, 2, 4, 5], "B": [0, 2, 3, 43, 2, 4, 0]}, index=[0, 0.1, 0.15, 0.11, 5, 7, 9])
    LOG.info("Raw data\n%s", db_raw)

    unifier = UnifySignal(target_sampling_rate_ms=0.1, target_tstart_ms=0, target_tend_ms=10)
    db_uni = unifier.unify(db=db_raw)
    LOG.info("Unified data\n%s", db_uni)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.DEBUG)
    LOG.info("Execute module tests")
    test()
    LOG.info("Module tests done")
