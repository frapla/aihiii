import logging
import math
import sys
from pathlib import Path, PosixPath, WindowsPath
from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
import polars as pl

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.Csv as csv_util
import src.utils.custom_log as custom_log
from src.data.fe_processing.IsoMme import IsoMme
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class InjuryCalculator:
    def __init__(
        self,
        data: Union[pd.DataFrame, WindowsPath, PosixPath, pl.DataFrame],
        mme: IsoMme,
        cfc: Literal["A", "B", "C", "D", "X"] = "A",
        scale2s: float = 0.001,
    ):
        """Calculate crash test dummy injury criteria from ISO style signals

        Args:
            data (Union[pd.DataFrame, WindowsPath, PosixPath]): object or path to table with shape (n time stamps, n channels)
            mme (IsoMme): object with ISO MME like naming convention
            cfc (Literal[&quot;A&quot;, &quot;B&quot;, &quot;C&quot;, &quot;D&quot;, &quot;X&quot;], optional): CFC filter class. Defaults to "A".
            scale2s (float, optional): scaling factor. Defaults to 0.001.
        """

        # get data
        if isinstance(data, pd.DataFrame):
            self.db: pd.DataFrame = data
        elif isinstance(data, pl.DataFrame):
            self.db: pl.DataFrame = data
        elif data.is_file():
            self.db: pd.DataFrame = csv_util.Csv(csv_path=data, log=LOG).read()
        else:
            LOG.error(f"File {data} does not exist")
            self.db: pd.DataFrame = pd.DataFrame()

        # determine sampling rate
        tm = self.db.index if isinstance(self.db, pd.DataFrame) else self.db[STR.time]

        self.dt: float = np.mean(np.diff(tm)) * scale2s  # [s]
        if self.dt is np.nan:
            self.dt = 0.001
            LOG.error("No valid time step found - set to %s", self.dt)

        # head
        self.a_head: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_head,
                dimension=mme._acc,
                direction=mme._res_direction,
                cfc=cfc,
            )
        )  # [g]

        # neck
        self.f_z: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_neck,
                dimension=mme._force,
                direction=mme._directions[2],
                cfc=cfc,
            ),
        )  # [kN]
        self.m_y: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_neck,
                dimension=mme._moment,
                direction=mme._directions[1],
                cfc=cfc,
            ),
        )  # [Nm]
        self.f_x: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_neck,
                dimension=mme._force,
                direction=mme._directions[0],
                cfc=cfc,
            ),
        )  # [kN]

        # chest
        self.a_chest: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_chest,
                dimension=mme._acc,
                direction=mme._res_direction,
                cfc=cfc,
            ),
        )  # [g]
        self.d_chest: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_chest,
                dimension=mme._displ,
                direction=mme._directions[0],
                cfc=cfc,
            ),
        )  # [mm]

        # femur
        self.f_femur_ri: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_femur_ri,
                dimension=mme._force,
                direction=mme._directions[-1],
                cfc=cfc,
            )
        )  # [kN]
        self.f_femur_le: np.ndarray | None = self.check(
            channel_name=mme.channel_name(
                sensor_loc=mme._loc_femur_le,
                dimension=mme._force,
                direction=mme._directions[-1],
                cfc=cfc,
            )
        )  # [kN]
        del self.db

        # init
        self.mme = mme
        self.injury_crit: Dict[str, float] = {}

    def check(self, channel_name: str) -> None | np.ndarray:
        """
        Check if channel name is in database
        :param channel_name: name of channel
        :return: channel signal values
        """
        if channel_name in self.db:
            LOG.debug(f"Use channel {channel_name}")
            if isinstance(self.db, pd.DataFrame):
                return self.db[channel_name].values
            else:
                return self.db[channel_name].to_numpy()
        else:
            LOG.warning(f"{channel_name} not in database")
            return None

    def calculate(self):
        """
        Calculate injury criteria
        """
        LOG.debug("Calculate injury criteria")
        self.__calculate_head()
        self.__calculate_neck()
        self.__calculate_chest()
        self.__calculate_femur_compression()
        self.__calculate_femur_tension()
        self.__calculate_femur_abs()

    def __calculate_head(self):
        """Calculate head related criteria"""
        LOG.debug("Calculate head related criteria")
        if self.a_head is None:
            LOG.warning("Skip calculation of HIC15, HIC36, and Head_a3ms")
        else:
            for t in [15, 36]:
                self.injury_crit[f"Head_HIC{t}"] = self.max_hic(hic_dt=t, signal=self.a_head)
            self.injury_crit["Head_a3ms"] = self.max_xms(xms_dt=3, signal=self.a_head)

    def __calculate_neck(self):
        """Calculate neck related criteria"""
        LOG.debug("Calculate neck related criteria")
        if self.f_z is None and self.m_y is None:
            LOG.warning("Skip calculation of Nij")
        else:
            self.injury_crit["Neck_Nij"] = self.max_nij(f_z=self.f_z, m_y=self.m_y)

        if self.f_z is None:
            LOG.warning("Skip calculation of Fz_Max_Compression and Fz_Max_Tensions")
        else:
            self.injury_crit["Neck_Fz_Max_Compression"] = max(abs(self.f_z[self.f_z < 0]), default=0)
            self.injury_crit["Neck_Fz_Max_Tension"] = max(self.f_z[self.f_z > 0], default=0)

        if self.m_y is None:
            LOG.warning("Skip calculation of My_Max")
        else:
            self.injury_crit["Neck_My_Max"] = max(abs(self.m_y), default=0)
            self.injury_crit["Neck_My_Extension"] = abs(min(self.m_y, default=0))
            self.injury_crit["Neck_My_Flexion"] = abs(max(self.m_y, default=0))

        if self.f_x is None:
            LOG.warning("Skip calculation of Fx_Shear_Max")
        else:
            self.injury_crit["Neck_Fx_Shear_Max"] = max(abs(self.f_x), default=0)

    def __calculate_chest(self):
        """Calculate chest related criteria"""
        LOG.debug("Calculate chest related criteria")
        if self.d_chest is None:
            LOG.warning("Skip calculation of Chest_Deflection")
        else:
            self.injury_crit["Chest_Deflection"] = max(abs(self.d_chest), default=0)

        if self.a_chest is None:
            LOG.warning("Skip calculation of Chest_a3ms")
        else:
            self.injury_crit["Chest_a3ms"] = self.max_xms(xms_dt=3, signal=self.a_chest)

        if self.a_chest is None:
            LOG.warning("Skip calculation of Chest_VC")
        else:
            sf = 1.3  # EURO NCAP for HIII
            d_const = {50: 229, 5: 187, 95: 254}[
                self.mme.dummy_percentile
            ]  # H.J. Mertz, Injury Risk Assessments Based on Dummy Responses
            v = (8 * (self.d_chest[3:-1] - self.d_chest[1:-3]) - self.d_chest[4:] + self.d_chest[:-4]) / (12 * self.dt * 1000)
            self.injury_crit["Chest_VC"] = max(sf * v * (self.d_chest[2:-2] / d_const), default=0)

    def __calculate_femur_compression(self):
        """Calculate femur max absolute force for compression (like Euro NCAP)"""
        LOG.debug("Calculate femur max absolute force")
        if self.f_femur_le is None or self.f_femur_ri is None:
            LOG.warning("Skip calculation of Femur_Fz_Max")
        else:
            val = min(
                [
                    min(self.f_femur_le, default=0),
                    min(self.f_femur_ri, default=0),
                ]
            )
            self.injury_crit["Femur_Fz_Max_Compression"] = abs(val) if val < 0 else 0

    def __calculate_femur_tension(self):
        """Calculate femur max absolute force for tension"""
        LOG.debug("Calculate femur max absolute force")
        if self.f_femur_le is None or self.f_femur_ri is None:
            LOG.warning("Skip calculation of Femur_Fz_Max")
        else:
            val = max(
                [
                    max(self.f_femur_le, default=0),
                    max(self.f_femur_ri, default=0),
                ]
            )
            self.injury_crit["Femur_Fz_Max_Tension"] = val if val > 0 else 0

    def __calculate_femur_abs(self):
        """Calculate femur max absolute force"""
        if "Femur_Fz_Max_Compression" in self.injury_crit and "Femur_Fz_Max_Tension" in self.injury_crit:
            self.injury_crit["Femur_Fz_Max"] = max(
                self.injury_crit["Femur_Fz_Max_Compression"], self.injury_crit["Femur_Fz_Max_Tension"]
            )
        else:
            LOG.warning("Skip calculation of Femur_Fz_Abs_Max")

    def __reshape2window(self, window: float, array: np.ndarray) -> np.ndarray:
        """
        Reshape given array

        Args:
            window (float): size of time window
            array (np.ndarray): to reformat, array of shape (n time stamps)

        Returns:
            np.ndarray: reshaped array (n, window length)
        """

        try:
            win_ele = math.ceil(window / self.dt)
        except ValueError:
            LOG.error("Invalid window size %s for dt %s", window, self.dt)

        # reformat to windows
        s = np.array([array[i : i + win_ele] for i in range(array.shape[0] - win_ele + 1)])

        return s

    def max_hic(self, hic_dt: float, signal: np.ndarray) -> float:
        """Calculate Head Injury Criterion (HIC)

        Args:
            hic_dt (float): evaluation time interval (typically 15 or 36) in [ms]
            signal (np.ndarray): acceleration in [g]

        Returns:
            float: HIC in [s * g**2.5]
        """
        hic = 0
        for hic_dt_sub in np.arange(self.dt, hic_dt * 0.001 + self.dt, self.dt):
            # time window, assume nearly constant sampling rate
            fact = hic_dt_sub / (hic_dt_sub**2.5)

            # reformat to windows
            reshaped = self.__reshape2window(window=hic_dt_sub, array=signal)

            # calculate HIC for all windows
            reshaped = np.trapz(reshaped, dx=self.dt, axis=1)
            hic = max(hic, (max(reshaped, default=0) ** 2.5) * fact)

        return hic

    def max_xms(self, xms_dt: float, signal: np.ndarray) -> float:
        """Calculate the mean average over given time interval

        Args:
            xms_dt (float): evaluation time interval (typically 3) in [ms]
            signal (np.ndarray): acceleration in [g]

        Returns:
            float: maximum average in [g]
        """
        # time window, assume nearly constant time step
        xms_dt *= 0.001

        # reformat to windows
        reshaped = self.__reshape2window(window=xms_dt, array=signal)

        # calculate average for each time window
        xms = reshaped.mean(axis=1)

        # get maximum
        xms_max = max(xms, default=0)

        return xms_max

    def max_nij(self, f_z: np.ndarray, m_y: np.ndarray) -> float:
        """Calculate total Neck Injury Criterion (Nij) with references from FMVSS208 for 50th percentile

        Args:
            f_z (np.ndarray): axial force [kN]
            m_y (np.ndarray): bending moment [Nm]

        Returns:
            float: Nij [1]
        """
        # intercept values
        # specified in FMVSS208 §6.6(a)(3) / §15.3.6(a)(3) for 50th, 5th in agreement with
        # Eppinger et al. (2000): Development of Improved Injury Criteria for the Assessment of Advanced Automotive Restraint Systems - II. NHTSA.
        # intercept for 95th for Eppinger et al.
        fzc_tension = {5: 4287, 50: 6806, 95: 8216}[self.mme.dummy_percentile]  # [N]
        fzc_compr = {5: 3880, 50: 6160, 95: 7440}[self.mme.dummy_percentile]  # [N]
        myc_flexion = {5: 155, 50: 310, 95: 415}[self.mme.dummy_percentile]  # [Nm]
        myc_extension = {5: 67, 50: 135, 95: 179}[self.mme.dummy_percentile]  # [Nm]

        f_int_tension = fzc_tension * 0.001  # [kN]
        f_int_compression = fzc_compr * 0.001  # [kN]
        my_flexion = myc_flexion
        my_extension = myc_extension  # [Nm]

        # separate parts
        f_z_compression = max(abs(f_z[f_z < 0]), default=0) / f_int_compression
        f_z_tension = max(f_z[f_z > 0], default=0) / f_int_tension

        m_y_extension = max(abs(m_y[m_y < 0]), default=0) / my_extension
        m_y_flexion = max(m_y[m_y > 0], default=0) / my_flexion

        nij = max([f_z_compression, f_z_tension]) + max([m_y_flexion, m_y_extension])

        return nij

    def save(self, case_dir: Path):
        """Save calculated criteria as dictionary to json

        Args:
            case_dir (Path): directory to save calculated criteria to
        """
        LOG.debug("Store injury criteria")
        try:
            idx = int(case_dir.stem[1:])
        except ValueError:
            idx = 0
        db = pd.DataFrame(self.injury_crit, index=[idx])
        out_path = case_dir / "injury_criteria.parquet"
        LOG.info("Save injury criteria to %s", out_path)
        db.to_parquet(out_path, index=True)
        # json_util.dump(obj=self.injury_crit, f_path=case_dir / "injury_criteria")


def test():
    b_path = Path(r"data\samples\binout\channels.csv.zip")
    mme = IsoMme(dummy_percentile=50, dummy_position="03", dummy_type="H3")
    calculator = InjuryCalculator(data=b_path, mme=mme)
    calculator.calculate()
    calculator.save(case_dir=b_path.parent)

    LOG.info("DONE")


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.DEBUG)
    test()
