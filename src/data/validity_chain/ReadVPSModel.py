import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr

src_dir = str(Path(__file__).absolute().parents[3])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.data.validity_chain.ReadFeModel import ReadFeModel
from src.utils.hash_file import hash_file

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

LOG: logging.Logger = logging.getLogger(__name__)


class ReadVPSModel(ReadFeModel):
    def __init__(self, in_dir: Path, out_dir: Path, dummy_v: str, dummy_perc: Literal[5, 50, 95]) -> None:
        """Read VPS Simulation data from erfh5

        Args:
            in_dir (Path): directory of VPS simulation
            out_dir (Path): directory to store results in
            dummy_v (str): version of dummy (e.g. "VI", "H3")
            dummy_perc (int): percentile
        """
        super().__init__(in_dir=in_dir, out_dir=out_dir, dummy_v=dummy_v, dummy_perc=dummy_perc)
        self.__allowed_dummy_versions: List[str] = ["VI", "H3"]

        if dummy_v in set(self.__allowed_dummy_versions):
            self.__dummy_version: str = dummy_v
        else:
            LOG.critical("Dummy version %s not supported - specify from: %s", dummy_v, self.__allowed_dummy_versions)

        self.__perc: str = f"{dummy_perc:02d}"
        self.sampling_rate_ms: float = 0.1
        self.target_tend_ms: int = 140
        self.out_file_stem: str = "extracted"

        # to read
        self.channels: List[str] = [
            "00COG00000VH00VEX",
            "00COG00000VH00VEY",
            "00COG00000VH00ACY",
            "00COG00000VH00ACX",
        ]
        for pos in ["01", "03"]:
            if pos == "03":
                self.channels.extend(
                    [
                        f"{pos}FEMRLE00{self.__dummy_version[:2]}{self.__perc}FOZ",
                        f"{pos}FEMRRI00{self.__dummy_version[:2]}{self.__perc}FOZ",
                        f"{pos}NECKUP00{self.__dummy_version[:2]}{self.__perc}FOZ",
                        f"{pos}NECKUP00{self.__dummy_version[:2]}{self.__perc}FOX",
                        f"{pos}NECKUP00{self.__dummy_version[:2]}{self.__perc}MOY",
                        f"{pos}CHST0000{self.__dummy_version[:2]}{self.__perc}DSX",  # chest deflection
                        f"{pos}BELTB000VH00DSR",
                        f"{pos}BELTB000VH00FOR",
                        f"{pos}BELTB300VH00FOR",
                        f"{pos}BELTB400VH00FOR",
                        f"{pos}BELTB500VH00FOR",
                        f"{pos}BELTB600VH00FOR",
                        f"{pos}BELTBUSLVH00DSR",
                        f"{pos}FAB00000VH00PRR",
                        f"{pos}FAB00000VH00VOR",
                        f"{pos}FAB00000VH00TPR",
                        f"{pos}FAB00000VH00IMR",
                        f"{pos}FAB00000VH00OMR",
                        f"{pos}FAB00000VH00TMR",
                    ]
                )

            for d in "XYZ":
                self.channels.extend(
                    [
                        f"{pos}SILFRONTVH00DS{d}",
                        f"{pos}SILREAR0VH00DS{d}",
                    ]
                )
                if pos == "03":
                    self.channels.extend(
                        [
                            f"{pos}HEAD0000{self.__dummy_version[:2]}{self.__perc}AC{d}",
                            f"{pos}HEADLOC0{self.__dummy_version[:2]}{self.__perc}DS{d}",
                            f"{pos}CHST0000{self.__dummy_version[:2]}{self.__perc}AC{d}",
                            f"{pos}PELV0000{self.__dummy_version[:2]}{self.__perc}AC{d}",
                            f"{pos}PELVLOC0{self.__dummy_version[:2]}{self.__perc}DS{d}",
                            f"{pos}CHSTLOC0{self.__dummy_version[:2]}{self.__perc}DS{d}",
                        ]
                    )

        self.__start_path: str = "/post/multistate/TIMESERIES1/multientityresults/"
        self.__const_path: str = "/post/constant/"
        self.__ch_field: str = "Channel"
        self.__time_field: str = "Time"
        self.__dimension_field: str = "DIM"

    def run(self) -> pd.DataFrame:
        """Get data from VPS simulation

        Returns:
            pd.DataFrame: data in shape (n_timestamps, n_channels), index are time stamps
        """
        LOG.info("Read VPS simulation data from %s", self.dir)
        # read
        raw_data = self.__read_erfh5()

        # process
        picked_data = self.__pick(db=raw_data)
        unified_data = self._unify(db=picked_data)
        filtered_data = self._filter_cfc(db=unified_data)
        with_resultants = self._add_resultants(db=filtered_data)
        with_locals = self._add_locals(db=with_resultants)
        self.data = with_locals.copy()

        # store
        self.data.index.name = "TIME"
        self.store(db_name="extracted")

        return self.data.copy()

    def __object_ids(self) -> Dict[str, Union[int, List[int]]]:
        """Storage of object IDs

        Returns:
            Dict[str, int]: keys are ISO MME (without filter and direction), values are object IDs in erfh5
        """
        obj_ids = {
            "00COG00000VH00AC": "Vehicle_cg",
            "00COG00000VH00VE": "Vehicle_cg",
            "01SILFRONTVH00DS": "B_PillarSillLeft",
            "01SILREAR0VH00DS": "RearSillDriver",
            "03SILFRONTVH00DS": "B_PillarSillRight",
            "03SILREAR0VH00DS": "RearSillPassenger",
            "03BELTB000VH00FO": "RETRACTOR_Force",
            "03BELTB000VH00DS": "RETRACTOR_Reel_InOut_Length",
            "03BELTB300VH00FO": "SEAT_BELT_FORCE_B3_66000001",
            "03BELTB400VH00FO": "SEAT_BELT_FORCE_B4_66000002",
            "03BELTB500VH00FO": "SEAT_BELT_FORCE_B5_66000003",
            "03BELTB600VH00FO": "SEAT_BELT_FORCE_B6_66000004",
            "03BELTBUSLVH00DS": "SLIPRING_Length",
            "03FAB00000VH00PR": "Airbag_Pressure",
            "03FAB00000VH00VO": "Airbag_Volume",
            "03FAB00000VH00TP": "Airbag_Temperature",
            "03FAB00000VH00IM": "Airbag_Inlet_Mass",
            "03FAB00000VH00OM": "Airbag_Outlet_Mass",
            "03FAB00000VH00TM": "Airbag_Mass",
        }

        if self.__dummy_version == self.__allowed_dummy_versions[0]:  # VIRTHUMAN
            dummy_ids = {
                f"03FEMRLE00VI{self.__perc}FO": 90000420,
                f"03FEMRRI00VI{self.__perc}FO": 90000430,
                f"03NECKUP00VI{self.__perc}FO": 90000121,
                f"03NECKUP00VI{self.__perc}MO": 90000121,
                f"03CHST0000VI{self.__perc}DS": 90001303,
                f"03HEAD0000VI{self.__perc}AC": "VH01_Head_COG",
                f"03CHST0000VI{self.__perc}AC": "VH01_T8_acc",
                f"03PELV0000VI{self.__perc}AC": "VH01_Pelvis_COG",
                f"03HEADLOC0VI{self.__perc}DS": "VH01_Head_COG",
                f"03CHSTLOC0VI{self.__perc}DS": "VH01_T8_acc",
                f"03PELVLOC0VI{self.__perc}DS": "VH01_Pelvis_COG",
            }
        else:
            LOG.critical("Dummy version %s not supported - specify from: %s", self.__dummy_version, self.__allowed_dummy_versions)

        obj_ids.update(dummy_ids)

        return obj_ids

    def _store_obj_path(self, ch_name: str) -> Tuple[str, str]:
        """Set path and channel related object ids

        Args:
            ch_name (str): channel name in extended 18 characters ISO MME format without CFC (e.g. 03HEAD0000H350ACX)

        Returns:
            Tuple[str, str]: erfh5 filed and channel related object ID
        """
        LOG.debug("Extract erfh5 path from %s", ch_name)
        # set erfh5 path
        if ch_name[2:6] == "BELT":
            if ch_name[6:10] == "BUSL":
                erfh5_field = "SLIPRING"
            elif ch_name[6:8] == "B0":
                erfh5_field = "RETRACTOR"
            else:
                erfh5_field = "SECTION"
        else:
            if (
                ch_name[-3:-1] in {"AC", "VE"}
                or (ch_name[10:12] == "VH" and ch_name[-3:-1] in {"AC", "VE", "DS"})
                or "LOC" in ch_name
            ):
                erfh5_field = ch_name[-3:-1]
            elif ch_name[-3:-1] == "DS":
                erfh5_field = "JOINT_DISPLACEMENT"
            elif ch_name[-3:-1] == "FO":
                erfh5_field = "JOINT_FORCE"
            elif ch_name[-3:-1] == "MO":
                erfh5_field = "JOINT_MOMENT"
            elif ch_name[2:5] == "FAB":
                erfh5_field = "AIRBAG"
            else:
                erfh5_field = "SECTION"

        # get related object ID
        eid = self.__object_ids()[ch_name[:-1]]

        LOG.debug("ERFH5 path for channel %s is %s with ID %s", ch_name, erfh5_field, eid)
        return erfh5_field, eid

    def __read_nodes(self, h5: h5py.File) -> Dict[str, xr.DataArray]:
        raw_data = {}
        # read NODE data
        dims = {"VE": "Velocity", "AC": "Acceleration", "DS": "COORDINATE"}
        no_hists = h5.get(f"{self.__const_path}/attributes/NODE/erfblock/title")
        if no_hists is None:
            no_hists = h5.get(f"{self.__const_path}/attributes/NODE/NODE/erfblock/title")
        no_hists = no_hists.asstr()
        for di in dims.keys():
            data = h5.get(f"{self.__start_path}NODE/{dims[di]}/ZONE1_set1/erfblock/res")
            time_f = h5.get(f"{self.__start_path}NODE/{dims[di]}/ZONE1_set1/erfblock/indexval")
            if data is None:
                data = h5.get(f"{self.__start_path}NODE/NODE/{dims[di]}/ZONE1_set1/erfblock/res")
                time_f = h5.get(f"{self.__start_path}NODE/NODE/{dims[di]}/ZONE1_set1/erfblock/indexval")
            raw_data[di] = xr.DataArray(
                data=data[:, :, :],
                coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, no_hists), (self.__dimension_field, list("XYZ"))],
            )

        return raw_data

    def __read_retractor(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}RETRACTOR/Retractor_Variables/ZONE1_set1/erfblock/res")
        time_f = h5.get(f"{self.__start_path}RETRACTOR/Retractor_Variables/ZONE1_set1/erfblock/indexval")
        chs = h5.get(f"{self.__const_path}variablegroup/Retractor_Variables/erfblock/varkey")
        if data is None:
            data = h5.get(f"{self.__start_path}RETRACTOR/RETRACTOR/Retractor_Variables/ZONE1_set1/erfblock/res")
            time_f = h5.get(f"{self.__start_path}RETRACTOR/RETRACTOR/Retractor_Variables/ZONE1_set1/erfblock/indexval")
            chs = h5.get(f"{self.__const_path}variablegroup/variablegroup/Retractor_Variables/erfblock/varkey")
        return xr.DataArray(
            data=data[:, 0, :],
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs.asstr())],
        )

    def __read_slipring(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}SLIPRING/Slipring_Variables/ZONE1_set1/erfblock/res")
        time_f = h5.get(f"{self.__start_path}SLIPRING/Slipring_Variables/ZONE1_set1/erfblock/indexval")
        chs = h5.get(f"{self.__const_path}variablegroup/Slipring_Variables/erfblock/varkey")
        uid = h5.get(f"{self.__const_path}identifiers/SLIPRING/erfblock/uid")
        if data is None:
            data = h5.get(f"{self.__start_path}SLIPRING/SLIPRING/Slipring_Variables/ZONE1_set1/erfblock/res")
            time_f = h5.get(f"{self.__start_path}SLIPRING/SLIPRING/Slipring_Variables/ZONE1_set1/erfblock/indexval")
            chs = h5.get(f"{self.__const_path}variablegroup/variablegroup/Slipring_Variables/erfblock/varkey")
            uid = h5.get(f"{self.__const_path}identifiers/identifiers/SLIPRING/erfblock/uid")
        db = xr.DataArray(
            data=data[:, :, :],
            coords=[(self.__time_field, time_f[:, 0]), ("ID", uid[:, 0]), (self.__ch_field, chs.asstr())],
        )
        return db.sel(ID=66000004)

    def __read_airbag(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}AIRBAG/BAG_VARIABLES/ZONE1_set1/erfblock/res")
        time_f = h5.get(f"{self.__start_path}AIRBAG/BAG_VARIABLES/ZONE1_set1/erfblock/indexval")
        chs = h5.get(f"{self.__const_path}variablegroup/BAG_VARIABLES/erfblock/varkey")
        if data is None:
            data = h5.get(f"{self.__start_path}AIRBAG/AIRBAG/BAG_VARIABLES/ZONE1_set1/erfblock/res")
            time_f = h5.get(f"{self.__start_path}AIRBAG/AIRBAG/BAG_VARIABLES/ZONE1_set1/erfblock/indexval")
            chs = h5.get(f"{self.__const_path}variablegroup/variablegroup/BAG_VARIABLES/erfblock/varkey")
        return xr.DataArray(
            data=data[:, 0, :],
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs.asstr())],
        )

    def __read_section(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}SECTION/Section_Force/ZONE1_set1/erfblock/res")
        time_f = h5.get(f"{self.__start_path}SECTION/Section_Force/ZONE1_set1/erfblock/indexval")
        chs = h5.get(f"{self.__const_path}attributes/SECTION/erfblock/title")
        if data is None:
            data = h5.get(f"{self.__start_path}SECTION/SECTION/Section_Force/ZONE1_set1/erfblock/res")
            time_f = h5.get(f"{self.__start_path}SECTION/SECTION/Section_Force/ZONE1_set1/erfblock/indexval")
            chs = h5.get(f"{self.__const_path}attributes/attributes/SECTION/erfblock/title")
        return xr.DataArray(
            data=np.linalg.norm(data, axis=2),
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs.asstr())],
        )

    def __read_joint_force(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}/JOINT/Force/res")
        chs = h5.get(f"{self.__start_path}/JOINT/Force/uid")
        time_f = h5.get(f"{self.__start_path}/JOINT/Force/indexval")
        if time_f is None:
            time_f = h5.get(f"{self.__start_path}SECTION/SECTION/Section_Force/ZONE1_set1/erfblock/indexval")

        return xr.DataArray(
            data=data[:, :, :],
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs), (self.__dimension_field, list("XYZ"))],
        )

    def __read_joint_moment(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}/JOINT/Moment_R/res")
        chs = h5.get(f"{self.__start_path}/JOINT/Moment_R/uid")
        time_f = h5.get(f"{self.__start_path}/JOINT/Moment_R/indexval")
        if time_f is None:
            time_f = h5.get(f"{self.__start_path}SECTION/SECTION/Section_Force/ZONE1_set1/erfblock/indexval")

        return xr.DataArray(
            data=data[:, :, 0],
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs)],
        )

    def __read_joint_displacement(self, h5: h5py.File) -> xr.DataArray:
        data = h5.get(f"{self.__start_path}/JOINT/Relative_Displacement/res")
        chs = h5.get(f"{self.__start_path}/JOINT/Relative_Displacement/uid")
        time_f = h5.get(f"{self.__start_path}/JOINT/Relative_Displacement/indexval")
        if time_f is None:
            time_f = h5.get(f"{self.__start_path}SECTION/SECTION/Section_Force/ZONE1_set1/erfblock/indexval")

        return xr.DataArray(
            data=np.linalg.norm(data, axis=2),
            coords=[(self.__time_field, time_f[:, 0]), (self.__ch_field, chs)],
        )

    def __read_erfh5(self) -> Dict[str, xr.DataArray]:
        """Read erfh5 from file system

        Returns:
            Dict[str, xr.DataArray]: field names are key
                values are tables of shape (individual n_timestamps, n_ids), index is time
        """
        LOG.info("Read erfh5 data from %s", self.dir)
        # init
        raw_data: Dict[str, xr.DataArray] = {}

        # prepare
        self.in_files = [x for x in self.dir.glob("*.erfh5")]
        if not self.in_files:
            LOG.critical("No .erfh5 file found in %s", self.dir)
        else:
            self.in_hash = hash_file(fpath=self.in_files)

        with h5py.File(self.in_files[0]) as h5:
            # read NODE data
            raw_data.update(self.__read_nodes(h5=h5))

            # read RETRACTOR data
            raw_data["RETRACTOR"] = self.__read_retractor(h5=h5)

            # read SECTION data
            raw_data["SECTION"] = self.__read_section(h5=h5)

            # read AIRBAG data
            raw_data["AIRBAG"] = self.__read_airbag(h5=h5)

            # read slipring data
            raw_data["SLIPRING"] = self.__read_slipring(h5=h5)

            # read JOINT data
            raw_data["JOINT_FORCE"] = self.__read_joint_force(h5=h5)
            raw_data["JOINT_MOMENT"] = self.__read_joint_moment(h5=h5)
            raw_data["JOINT_DISPLACEMENT"] = self.__read_joint_displacement(h5=h5)

        return raw_data

    def __pick(self, db: Dict[str, xr.DataArray]) -> List[pd.DataFrame]:
        """Select relevant data from raw erfh5 data

        Args:
            db (Dict[tuple, pd.DataFrame]): keys are storing path in erfh5
                values are tables of shape (individual n_timestamps, n_ids), index is time

        Returns:
            List[pd.DataFrame]: tables of shape (individual n_timestamps, 1), index is time
        """
        LOG.info("Try to get %s channels from h5 data", len(self.channels))
        # init
        dbs = []
        chs_new = self.channels.copy()

        # process channels
        for channel in self.channels:
            # get id
            erfh5_field, eid = self._store_obj_path(ch_name=channel)
            if eid is not None and (isinstance(eid, int) or eid.startswith("VH01")):
                cannot_find_it = True
                trials = 0
                while cannot_find_it and trials < 10:
                    try:
                        _ = db[erfh5_field].sel({self.__ch_field: eid})
                        cannot_find_it = False
                    except KeyError:
                        LOG.warning("No data for channel %s, %s found - number up :)", channel, eid)
                        trials += 1
                        if isinstance(eid, int):
                            eid += 100000
                        else:
                            eid = eid[:2] + f"{int(eid[2:4])+1:02d}" + eid[4:]
                        LOG.warning("Try to find %s", eid)

            try:
                LOG.info("Get data for channel %s with erfh5_field %s and eid %s", channel, erfh5_field, eid)
                if channel[-1] != "R" and channel[-3:-1] != "MO":
                    if "CHST0000" in channel and channel[-3:-1] == "DS":
                        data = db[erfh5_field].sel({self.__ch_field: eid})  # chest deflection
                    else:
                        data = db[erfh5_field].sel({self.__dimension_field: channel[-1], self.__ch_field: eid})
                else:
                    data = db[erfh5_field].sel({self.__ch_field: eid})
                dbs.append(pd.DataFrame({channel: data.to_series()}))
                LOG.info("Data for channel %s has shape %s", channel, data.shape)
            except KeyError:
                LOG.debug(
                    "Data for channel %s with erfh5_field %s and eid %s:\n %s",
                    channel,
                    erfh5_field,
                    eid,
                    db[erfh5_field].to_series(),
                )
                LOG.warning("No data for channel %s, %s found - SKIP and Remove from channel list", channel, eid)
                chs_new = [ch for ch in chs_new if ch != channel]
                continue

        # update channels
        self.channels = chs_new

        LOG.info("%s channels selected", len(dbs))
        return dbs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        help="Path to directory with single erfh5 file",
    )
    parser.add_argument(
        "--out",
        required=False,
        default=None,
        type=Optional[Path],
        help="Path to directory to store generated data in",
    )
    parser.add_argument(
        "--dummy",
        required=True,
        type=str,
        help="Dummy type (e.g. VI, H3)",
    )
    parser.add_argument(
        "--percentile",
        required=True,
        type=int,
        help="Dummy Percentile (e.g. 5, 50, 95)",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=logging.INFO,
        type=int,
        help="Log level (default is %(default)s)",
    )
    args = parser.parse_args()

    # init logger
    custom_log.init_logger(log_lvl=args.log_lvl)
    LOG.info("START Read VPS Model")
    reader = ReadVPSModel(
        in_dir=args.directory,
        out_dir=args.directory if args.out is None else args.out,
        dummy_v=args.dummy,
        dummy_perc=args.percentile,
    )
    db = reader.run()

    LOG.info("Data has shape %s", db.shape)
    LOG.debug("Data:\n%s", db)

    LOG.info("DONE")


if __name__ == "__main__":
    run()
