import argparse
import sys
from collections import defaultdict
import logging
from pathlib import Path
from typing import Literal, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

sys.path.append(str(Path(__file__).absolute().parents[3]))

import src.utils.custom_log as custom_log
from src.data.fe_processing.IsoMme import IsoMme
from src.data.validity_chain.ReadGeneral import ReadGeneral

LOG: logging.Logger = logging.getLogger(__name__)


class ReadReportData(ReadGeneral):
    def __init__(self, in_dir: Path, out_dir: Union[Path, None] = None) -> None:
        """Parse CSV files in directory to unified and ISO MME style named database

        Args:
            in_dir (Path): directory with CSV files
            out_dir Union[Path, None], optional): directory to store data in. Defaults to None.
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        super().__init__(in_dir=in_dir, out_dir=out_dir)

        self.channels_to_flip: Set[str] = {
            "00COG00000VH00ACXD",
            "00COG00000VH00VEXD",
            "01HEAD0000TH50ACXD",
            "01HEAD0000TH50ACZD",
            "01PELV0000TH50ACYD",
            "01PELV0000TH50ACZD",
            "03HEAD0000TH50ACXD",
            "03HEAD0000TH50ACZD",
            "00COG00000VH00VEYD",
        }

    def run(self):
        """Parse csv files to database"""
        # get data from files
        data = self.__get_data()

        # add resultants
        self.data = self.__add_resultant(db=data)

    def __get_data(self) -> pd.DataFrame:
        contents = defaultdict(list)
        for csv in self.dir.glob("*.csv"):
            LOG.debug("Read %s", csv)
            # get info
            iso, source, loadcase = self.__parse_csv_name(csv=csv)

            # read file
            self.in_files.append(csv)
            csv_data = pd.read_csv(csv, sep=";", decimal=",", names=["Time", "Value"])
            csv_data = csv_data.sort_values(by="Time")

            # unify
            x_new, y_new = self.__interpolate_data(x_old=csv_data["Time"].to_numpy(), y_old=csv_data["Value"].to_numpy())

            # convert to unit system [mm, ms, kg], acceleration kept in [g]
            x_new *= 1000  # [s] -> [ms]
            if iso[-4:-2] == "FO":
                y_new *= 0.001  # [N] -> [kN]

            # adjust directions to fit own simulations
            if iso in self.channels_to_flip:
                y_new *= -1

            # store
            contents["Value"].extend(y_new)
            contents["Time"].extend(x_new)

            # add information
            contents["Channel"].extend([iso] * y_new.shape[0])
            contents["Source"].extend([source] * y_new.shape[0])
            contents["Case"].extend([self.standard_cases[loadcase]] * y_new.shape[0])

        # merge data
        return pd.DataFrame(contents)

    def __interpolate_data(self, x_old: np.ndarray, y_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ensure monotonous time stamps
        for i in range(1, x_old.shape[0]):
            if x_old[i] == x_old[i - 1]:
                x_old[i] += 0.000001
            elif x_old[i] < x_old[i - 1]:
                x_old[i] += 2 * (x_old[i - 1] - x_old[i])

        # interpolate
        interpolator = Akima1DInterpolator(x=x_old, y=y_old)
        x_new = np.arange(0.001, 0.14, 0.001)
        y_new = interpolator(x_new)

        return x_new, y_new

    def __add_resultant(self, db: pd.DataFrame) -> pd.DataFrame:
        contents = defaultdict(list)
        for case in db["Case"].unique():
            for source in db["Source"].unique():
                filt: pd.DataFrame = db[db["Case"].eq(case) & db["Source"].eq(source)].copy()
                db_col = []
                for channel in filt["Channel"].unique():
                    x = filt["Time"][filt["Channel"].eq(channel)].to_numpy()
                    y = filt["Value"][filt["Channel"].eq(channel)].to_numpy()
                    db_col.append(pd.Series(y, index=x, name=channel))

                db_col = pd.DataFrame(db_col).T
                self.channels = db_col.columns.to_list()
                db_res = self._add_resultants(db=db_col)
                for channel in db_res.columns:
                    contents["Value"].extend(db_res[channel])
                    contents["Time"].extend(db_res.index)
                    contents["Channel"].extend([channel] * db_res.shape[0])
                    contents["Source"].extend([source] * db_res.shape[0])
                    contents["Case"].extend([case] * db_res.shape[0])

        return pd.DataFrame(contents)

    def __parse_csv_name(self, csv: Path) -> Tuple[str, str, str]:
        """Generate ISO MME style naming from csv file name

        Args:
            csv (Path): path to csv file

        Returns:
            Tuple[str, str, str]: channel ISO name, data source name, loadcase name
        """
        name_parts = csv.stem.split("_")
        LOG.debug("Name parts: %s", name_parts)

        # decompose
        loadcase: Literal["full_frontal_56kmh", "oblique_left_90kmh", "oblique_right_90kmh"] = "_".join(name_parts[:3])
        channel: Literal[
            "belt_upper_force",
            "cog_acceleration_x",
            "cog_acceleration_y",
            "cog_velocity_x",
            "cog_velocity_y",
            "femur_force_left",
            "femur_force_right",
            "front_dynamic_crush",
            "head_acceleration_res",
            "head_acceleration_x",
            "head_acceleration_y",
            "head_acceleration_z",
            "pelvis_acceleration_x",
            "pelvis_acceleration_y",
            "pelvis_acceleration_z",
        ] = "_".join(name_parts[4:-1])

        # prepare ISO MME
        mme = self.__get_iso_mme(location=name_parts[3], loadcase=loadcase)

        # generate channel name
        ch_name = mme.channel_name(
            sensor_loc=self.__get_sensor_location(mme=mme, channel=channel),
            dimension=self.__get_dimension(mme=mme, channel=channel),
            direction=self.__get_direction(mme=mme, channel=channel),
            cfc=60,
        )
        LOG.debug("Channel ISO name: %s", ch_name)

        return ch_name, f"{name_parts[-1]} NHTSA", loadcase

    def __get_iso_mme(
        self,
        location: Literal["vehicle", "driver", "passenger"],
        loadcase: Literal["full_frontal_56kmh", "oblique_left_90kmh", "oblique_right_90kmh"],
    ) -> IsoMme:
        """Init the ISO MME Module

        Args:
            location (Literal[&quot;vehicle&quot;, &quot;driver&quot;, &quot;passenger&quot;]): location in vehicle
            loadcase (Literal[&quot;full_frontal_56kmh&quot;, &quot;oblique_left_90kmh&quot;, &quot;oblique_right_90kmh&quot;]): crash loadcase

        Returns:
            IsoMme: ISO MME Module
        """
        location = {"vehicle": "00", "driver": "01", "passenger": "03"}[location]

        if location == "00":
            dummy_type = "VH"
            dummy_percentile = 0
        else:
            if loadcase.startswith("oblique"):
                dummy_type = "TH"
                dummy_percentile = 50
            else:
                dummy_type = "H3"
                if location == "driver":
                    dummy_percentile = 50
                else:
                    dummy_percentile = 5

        return IsoMme(dummy_type=dummy_type, dummy_percentile=dummy_percentile, dummy_position=location)

    def __get_dimension(self, mme: IsoMme, channel: str) -> str:
        """Parse name to physical MME dimension

        Args:
            mme (IsoMme): MME object
            channel (str): channel description

        Returns:
            str: physical MME dimension
        """
        if "acceleration" in channel:
            dimension = mme._acc
        elif "velocity" in channel:
            dimension = mme._vel
        elif "force" in channel:
            dimension = mme._force
        elif "dynamic_crush" in channel:
            dimension = mme._displ
        else:
            dimension = ""

        return dimension

    def __get_direction(self, mme: IsoMme, channel: str) -> str:
        """Parse name to physical MME direction

        Args:
            mme (IsoMme): MME object
            channel (str): channel description

        Returns:
            str: physical MME direction
        """
        dirs = {"x": mme._directions[0], "y": mme._directions[1], "z": mme._directions[2], "res": mme._res_direction}
        if channel.split("_")[-1] in dirs:
            direction = dirs[channel.split("_")[-1]]
        else:
            direction = mme._res_direction

        return direction

    def __get_sensor_location(self, mme: IsoMme, channel: str) -> str:
        """Parse name to MME style sensor location

        Args:
            mme (IsoMme): MME object
            channel (str): channel description

        Returns:
            str: MME style sensor location
        """
        locs = {"belt": mme._loc_belt_b3, "pelvis": mme._loc_pelvis, "head": mme._loc_head}
        if channel.split("_")[0] in locs:
            sensor_loc = locs[channel.split("_")[0]]
        elif channel.startswith("femur"):
            if channel.endswith("left"):
                sensor_loc = mme._loc_femur_le
            else:
                sensor_loc = mme._loc_femur_ri
        elif channel.startswith("cog"):
            sensor_loc = mme._loc_cog
        elif channel.startswith("front"):
            sensor_loc = mme._loc_front
        else:
            sensor_loc = ""

        return sensor_loc


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        help="Path to directory with CSV files",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to directory to store generated data in",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=10,
        type=int,
        help="Log level (default is %(default)s)",
    )
    args = parser.parse_args()

    custom_log.init_logger(log_lvl=args.log_lvl)

    # run
    data = ReadReportData(in_dir=args.directory, out_dir=args.out)
    data.run()
    data.store(db_name="report_data")


if __name__ == "__main__":
    run()
