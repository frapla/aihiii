import argparse
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import scipy

src_dir = str(Path(__file__).absolute().parents[3])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.data.validity_chain.ReadGeneral import ReadGeneral
from src.utils.CfCFilter import CfCFilter
from src.utils.hash_file import hash_file
from src.utils.local_displacement import get_displ_along_axis
from src.utils.ReadBinout import ReadBinout
from src.utils.UnifySignal import UnifySignal
from src.utils.UnitConverter import UnitConverter

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

LOG: logging.Logger = logging.getLogger(__name__)


class ReadFeModel(ReadGeneral):
    def __init__(self, in_dir: Path, out_dir: Path, dummy_v: str, dummy_perc: Literal[5, 50, 95] = 50) -> None:
        """Read LS-Dyna Simulation data from Binout

        Args:
            in_dir (Path): directory of LS-Dyna simulation
            out_dir (Path): directory to store results in
            dummy_v (str): version of dummy (e.g. TH2.1, TH2.7, H3Rigid)
        """
        super().__init__(in_dir=in_dir, out_dir=out_dir)
        self.__allowed_dummy_versions: List[str] = ["TH2.1", "TH2.7", "H3Rigid", "VI", "H3"]

        if dummy_v in set(self.__allowed_dummy_versions):
            self.__dummy_version: str = dummy_v
        else:
            LOG.critical("Dummy version %s not supported - specify from: %s", dummy_v, self.__allowed_dummy_versions)
        self.__perc: str = f"{dummy_perc:02d}"
        self.sampling_rate_ms: float = 0.1
        self.target_tend_ms: int = 140
        self.out_file_stem: str = "extracted"

        # unit conversions
        self.__conv_facts: UnitConverter = UnitConverter(unit_mass="t", unit_length="mm", unit_time="s")
        self.__conv_facts_mme: Dict[str, float] = {
            "AC": self.__conv_facts.acceleration2g(),
            "DS": self.__conv_facts.length2mm(),
            "VE": self.__conv_facts.velocity2ms(),
            "FO": self.__conv_facts.force2kn(),
            "MO": self.__conv_facts.moment2nm(),
            "PR": self.__conv_facts.pressure2kpa(),
            "VO": self.__conv_facts.volume2l(),
            "TP": self.__conv_facts.dummy(),
            "IM": self.__conv_facts.mass2g(),
            "OM": self.__conv_facts.mass2g(),
            "TM": self.__conv_facts.mass2g(),
        }

        # to read
        self.channels: List[str] = [
            "00COG00000VH00VEX",
            "00COG00000VH00VEY",
            "00COG00000VH00ACY",
            "00COG00000VH00ACX",
        ]
        for pos in ["01", "03"]:
            self.channels.extend(
                [
                    f"{pos}FEMRLE00{self.__dummy_version[:2]}50FOR",
                    f"{pos}FEMRRI00{self.__dummy_version[:2]}50FOR",
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
                        f"{pos}HEAD0000{self.__dummy_version[:2]}50AC{d}",
                        f"{pos}HEADLOC0{self.__dummy_version[:2]}50DS{d}",
                        f"{pos}CHST0000{self.__dummy_version[:2]}50AC{d}",
                        f"{pos}CHSTLOC0{self.__dummy_version[:2]}50DS{d}",
                        f"{pos}PELV0000{self.__dummy_version[:2]}50AC{d}",
                        f"{pos}PELVLOC0{self.__dummy_version[:2]}50DS{d}",
                    ]
                )

    def run(self) -> pd.DataFrame:
        """Get data from LS-Dyna simulation

        Returns:
            pd.DataFrame: data in shape (n_timestamps, n_channels), index are time stamps
        """
        # read
        raw_data = self.__read_binout()

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
            Dict[str, int]: keys are ISO MME (without filter and direction), values are object IDs in binout
        """
        obj_ids = {
            "00COG00000VH00AC": 82000066,
            "00COG00000VH00VE": 82000066,
            "01SILFRONTVH00DS": 82000028,
            "01SILREAR0VH00DS": 82000012,
            "03SILFRONTVH00DS": 82000020,
            "03SILREAR0VH00DS": 82000004,
            "01BELTB000VH00FO": 65000001,
            "01BELTB000VH00DS": 65000001,
            "01BELTB300VH00FO": 65000001,
            "01BELTB400VH00FO": 65000002,
            "01BELTB500VH00FO": 65000003,
            "01BELTB600VH00FO": 65000004,
            "01BELTBUSLVH00DS": 65000002,
            "03BELTB000VH00FO": 66000001,
            "03BELTB000VH00DS": 66000001,
            "03BELTB300VH00FO": 66000001,
            "03BELTB400VH00FO": 66000002,
            "03BELTB500VH00FO": 66000003,
            "03BELTB600VH00FO": 66000004,
            "03BELTBUSLVH00DS": 66000002,
        }

        if self.__dummy_version == self.__allowed_dummy_versions[0]:  # TH 2.1
            dummy_ids = {
                "01FEMRLE00TH50FO": 617756,
                "01FEMRRI00TH50FO": 517756,
                "01HEAD0000TH50AC": 108557,
                "01PELV0000TH50AC": 836580,
                "03FEMRLE00TH50FO": 68617757,
                "03FEMRRI00TH50FO": 68517757,
                "03HEAD0000TH50AC": 68108558,
                "03PELV0000TH50AC": 68836581,
            }
        elif self.__dummy_version == self.__allowed_dummy_versions[1]:  # TH 2.7
            dummy_ids = {
                "01FEMRLE00TH50FO": 67620100,
                "01FEMRRI00TH50FO": 67520100,
                "01HEAD0000TH50AC": 67100100,
                "01CHST0000TH50AC": 67700300,
                "01PELV0000TH50AC": 67800100,
                "01HEADLOC0TH50DS": 67100100,
                "01CHSTLOC0TH50DS": 67700300,
                "01PELVLOC0TH50DS": 67800100,
                "01FAB00000VH00PR": 1000001,
                "03FEMRLE00TH50FO": 68620100,
                "03FEMRRI00TH50FO": 68520100,
                "03HEAD0000TH50AC": 68100100,
                "03CHST0000TH50AC": 68700300,
                "03PELV0000TH50AC": 68800100,
                "03HEADLOC0TH50DS": 68100100,
                "03CHSTLOC0TH50DS": 68700300,
                "03PELVLOC0TH50DS": 68800100,
            }
            for bb in (
                "03FAB00000VH00PR",
                "03FAB00000VH00VO",
                "03FAB00000VH00TP",
                "03FAB00000VH00IM",
                "03FAB00000VH00OM",
                "03FAB00000VH00TM",
            ):
                dummy_ids[bb] = [69000001, 64000001, 67000001]
            for bb in (
                "01FAB00000VH00PR",
                "01FAB00000VH00VO",
                "01FAB00000VH00TP",
                "01FAB00000VH00IM",
                "01FAB00000VH00OM",
                "01FAB00000VH00TM",
            ):
                dummy_ids[bb] = [69000001, 64000001, 67000001]  # random
        elif self.__dummy_version == self.__allowed_dummy_versions[2]:  # H3Rigid
            dummy_ids = {
                "01FEMRLE00H350FO": 67000024,
                "01FEMRRI00H350FO": 67000025,
                "01HEAD0000H350AC": 67000001,
                "01CHST0000H350AC": 67000736,
                "01PELV0000H350AC": 67002091,
                "01HEADLOC0H350DS": 67000001,
                "01CHSTLOC0H350DS": 67000736,
                "01PELVLOC0H350DS": 67002091,
                "01FAB00000VH00PR": [1000001, 63000001],
                "03FEMRLE00H350FO": 68000024,
                "03FEMRRI00H350FO": 68000025,
                "03HEAD0000H350AC": 68000001,
                "03CHST0000H350AC": [68000736, 68001787],
                "03PELV0000H350AC": [68002091, 68003304],
                "03HEADLOC0H350DS": 68000001,
                "03CHSTLOC0H350DS": [68000736, 68001787],
                "03PELVLOC0H350DS": [68002091, 68003304],
            }
            for bb in (
                "03FAB00000VH00PR",
                "03FAB00000VH00VO",
                "03FAB00000VH00TP",
                "03FAB00000VH00IM",
                "03FAB00000VH00OM",
                "03FAB00000VH00TM",
            ):
                dummy_ids[bb] = [69000001, 64000001, 67000001]
            for bb in (
                "01FAB00000VH00PR",
                "01FAB00000VH00VO",
                "01FAB00000VH00TP",
                "01FAB00000VH00IM",
                "01FAB00000VH00OM",
                "01FAB00000VH00TM",
            ):
                dummy_ids[bb] = [69000001, 64000001, 67000001]  # random !!

        obj_ids.update(dummy_ids)

        return obj_ids

    def _store_obj_path(self, ch_name: str) -> Tuple[str, Union[int, List[int]]]:
        """Set path and channel related object ids

        Args:
            ch_name (str): channel name in extended 18 characters ISO MME format without CFC (e.g. 03HEAD0000H350ACX)

        Returns:
            Tuple[str, int]: binout path (multi components) and channel related object ID
        """
        LOG.debug("Extract binout path from %s", ch_name)
        # set binout path
        dimension = {"AC": "acceleration", "VE": "velocity", "DS": "coordinate"}
        if ch_name[-3:-1] in dimension and "BELT" not in ch_name:
            binout_path = ("nodout", f"{ch_name[-1].lower()}_{dimension[ch_name[-3:-1]]}")
        elif "BELT" in ch_name:
            if "BELTBUSL" in ch_name:
                binout_path = ("sbtout", "ring_slip")
            elif "BELTB0" in ch_name:
                binout_path = ("sbtout", "retractor_force" if "FO" in ch_name else "retractor_pull_out")
            else:
                binout_path = ("secforc", "total_force")
        elif "FAB" in ch_name:
            scnd_fields = {
                "PR": "pressure",
                "VO": "volume",
                "TP": "gas_temp",
                "IM": "dm_dt_in",
                "OM": "dm_dt_out",
                "TM": "total_mass",
            }
            binout_path = ("abstat", scnd_fields[ch_name[-3:-1]])
        else:
            if self.__dummy_version.startswith("TH"):
                binout_path = ("elout", "beam", "axial")
            else:
                binout_path = ("jntforc", "joints", "z_force")

        # get related object ID
        eid = self.__object_ids()[ch_name[:-1]]

        LOG.debug("Binout path for channel %s is %s with ID %s", ch_name, binout_path, eid)
        return binout_path, eid

    def _conv_factor(self, ch_name: str) -> float:
        """Get unit conversion factor from extended ISO MME format

        Args:
            ch_name (str): channel name in extended 18 characters ISO MME format without CFC (e.g. 03HEAD0000H350ACX)

        Returns:
            float: unit conversion factor
        """

        # airbag system have in dyna independent unit system
        conv = self.__conv_facts_mme[ch_name[-3:-1]]

        LOG.debug("Unit conversion factor for %s is %s", ch_name, conv)
        return conv

    def __read_binout(self) -> Dict[tuple, pd.DataFrame]:
        """Read binout from file system

        Returns:
            Dict[tuple, pd.DataFrame]: keys are storing path in Binout
                values are tables of shape (individual n_timestamps, n_ids), index is time
        """
        LOG.info("Read binout data from %s", self.dir)
        # init
        paths: Dict[str, List[Union[int, List[int]]]] = defaultdict(list)
        raw_data: Dict[tuple, pd.DataFrame] = {}

        # collect reading tasks
        for channel in self.channels:
            b_path, eid = self._store_obj_path(ch_name=channel)
            paths[b_path].append(eid)

        # read from binout
        with ReadBinout(sim_dir=self.dir) as binout:
            # read
            self.in_files = [Path(x) for x in sorted(binout.binout.filelist)]
            self.in_hash = hash_file(fpath=self.in_files)

            # store
            for binout_path in paths:
                LOG.debug("Get %s from binout", binout_path)
                if binout_path[0] == "sbtout":
                    data = binout.binout.read(*binout_path)
                    time_stamps = binout.binout.read("sbtout", "time")
                    obj_ids = binout.binout.read("sbtout", "slipring_ids" if binout_path[1] == "ring_slip" else "retractor_ids")
                    raw_data[binout_path] = pd.DataFrame(data, index=time_stamps, columns=obj_ids)
                elif binout_path[0] == "abstat":
                    data = binout.binout.read(*binout_path)
                    time_stamps = binout.binout.read("abstat", "time")
                    obj_ids = binout.binout.read("abstat", "ids")
                    raw_data[binout_path] = pd.DataFrame(data, index=time_stamps, columns=obj_ids)
                else:
                    try:
                        # read directly as DataFrame
                        raw_data[binout_path] = binout.binout.as_df(*binout_path)
                    except (IndexError, ValueError):
                        # incompatible format - create DataFrame
                        LOG.debug("Treat wrong stored ids")

                        # read
                        data = binout.binout.read(*binout_path)
                        time_stamps = binout.binout.read(*binout_path[:2], "time")
                        obj_ids = binout.binout.read(*binout_path[:2], "ids")[0]

                        # store
                        raw_data[binout_path] = pd.DataFrame(data, index=time_stamps, columns=obj_ids)

                LOG.debug("Added data with shape %s from path %s", raw_data[binout_path].shape, binout_path)

                # store
                raw_data[binout_path].columns = [int(col) for col in raw_data[binout_path].columns]

        return raw_data

    def __pick(self, db: Dict[tuple, pd.DataFrame]) -> List[pd.DataFrame]:
        """Select relevant data from raw binout data

        Args:
            db (Dict[tuple, pd.DataFrame]): keys are storing path in Binout
                values are tables of shape (individual n_timestamps, n_ids), index is time

        Returns:
            List[pd.DataFrame]: tables of shape (individual n_timestamps, 1), index is time
        """
        LOG.info("Try to get %s channels from Binout data", len(self.channels))
        # init
        dbs = []
        chs_new = self.channels.copy()

        # process channels
        for channel in self.channels:
            # get id
            b_path, eid = self._store_obj_path(ch_name=channel)

            # filter
            found_ch = False
            if isinstance(eid, list):
                for eeid in eid:
                    if eeid in db[b_path]:
                        LOG.debug("Add channel %s", channel)
                        d = db[b_path][eeid]
                        d_time = d.index
                        if channel[-3:-1] in {"IM", "OM"}:
                            d = scipy.integrate.cumulative_trapezoid(
                                y=d.values,
                                x=d.index,
                                initial=d.values[0],
                            )
                        dbs.append(pd.DataFrame({channel: d}, index=d_time))
                        found_ch = True
                        break
            else:
                if eid in db[b_path]:
                    LOG.debug("Add channel %s", channel)
                    dbs.append(pd.DataFrame({channel: db[b_path][eid]}))
                    found_ch = True

            if not found_ch:
                LOG.warning("No data for channel %s found - SKIP and Remove from channel list", channel)
                chs_new = [ch for ch in chs_new if ch != channel]

        # update channels
        self.channels = chs_new

        LOG.info("%s channels selected", len(dbs))
        return dbs

    def _unify(self, db: List[pd.DataFrame]) -> pd.DataFrame:
        """Unify channels on same sampling rate and end time

        Args:
            db (List[pd.DataFrame]): tables of shape (individual n_timestamps, 1), index is time

        Returns:
            pd.DataFrame: data table of shape (n_timestamps, n_channels), index is time
        """
        LOG.info("Unify %s channels on sampling rate %sms and end time %sms", len(db), self.sampling_rate_ms, self.target_tend_ms)

        # init
        db_col: List[pd.DataFrame] = []
        unifier = UnifySignal(target_sampling_rate_ms=self.sampling_rate_ms, target_tend_ms=self.target_tend_ms)

        # unify
        for channel in db:
            ch = channel.copy()

            if "FAB00000VH00PR" in ch.columns[0]:
                if ch.columns[0].startswith("01") or ch.loc[ch.index[0], ch.columns[0]] < 0.05:
                    conv = 10
                    offset = 0
                else:
                    conv = 10
                    offset = -1.01325
                ch *= conv
                ch += offset
                LOG.warning("Override conversion factor for %s to %s and offset by %s", ch.columns, conv, offset)
            else:
                ch *= self._conv_factor(ch_name=ch.columns[0])
                if (
                    self.__dummy_version == "VI"
                    and ch.columns[0][:2] in {"01", "03"}
                    and ch.columns[0][2:6] in {"HEAD", "CHST", "PELV"}
                    and ch.columns[0][16] in {"Y", "Z"}
                ):
                    # VIRTHUMAN y and z seems rotated to HIII
                    ch *= -1
                    LOG.warning("Flip %s", ch.columns)
            ch.index *= self.__conv_facts.time2ms()
            ch = unifier.unify(db=ch)
            db_col.append(ch)

        # assemble
        db_uni = pd.concat(db_col, axis=1)

        LOG.info("Data unified to shape %s", db_uni.shape)
        return db_uni

    def _filter_cfc(self, db: pd.DataFrame) -> pd.DataFrame:
        """Filter channels with CFC filter

        Args:
            db (pd.DataFrame): data table of shape (n_timestamps, n_channels), index is time

        Returns:
            pd.DataFrame: data table of shape (n_timestamps, n_filters*n_channels), index is time
        """
        LOG.info("Filter data with filter CFC %s with shape %s", self.cfcs, db.shape)

        # get values
        sampling_rate_s = self.sampling_rate_ms / self.__conv_facts.time2ms()
        db_raw = db[self.channels].values

        # apply filter

        data_filtered = []
        cfc_filter = CfCFilter()
        for cfc in self.cfcs:
            db_cfc = cfc_filter.filter(tsp=sampling_rate_s, signal=db_raw, cfc=cfc)

            # reformat
            db_cfc = pd.DataFrame(db_cfc, columns=self.channels, index=db.index)
            LOG.info("Data filtered by CFC%s with shape %s", cfc, db_cfc.shape)

            # store
            data_filtered.append(db_cfc.rename(columns={ch: f"{ch}{self.cfc_names[cfc]}" for ch in db_cfc.columns}))

        data_filtered = pd.concat(data_filtered, axis=1)
        self.channels = sorted(data_filtered.columns)
        LOG.info("Combine %s Filter classes to shape %s", len(self.cfcs), data_filtered.shape)
        LOG.debug("Channels are: %s", list(data_filtered.columns))
        return data_filtered

    def _add_locals(self, db: pd.DataFrame) -> pd.DataFrame:
        """Add local coordinates to data

        Args:
            db (pd.DataFrame): data table of shape (n_timestamps, n_channels), index is time

        Returns:
            pd.DataFrame: data table of shape (n_timestamps, n_channels), index is time
        """
        LOG.info("Add local coordinates to data with shape %s", db.shape)

        db_new = db.copy()

        for cfc_num in self.cfcs:
            cfc = self.cfc_names[cfc_num]
            # decide node set
            dr_side = np.linalg.norm(
                db[[f"01SILFRONTVH00DS{d}{cfc}" for d in "XYZ"]] - db[[f"01SILREAR0VH00DS{d}{cfc}" for d in "XYZ"]]
            ).max()
            pa_side = np.linalg.norm(
                db[[f"03SILFRONTVH00DS{d}{cfc}" for d in "XYZ"]] - db[[f"03SILREAR0VH00DS{d}{cfc}" for d in "XYZ"]]
            ).max()
            if dr_side < pa_side:
                root, dir_x, dir_y = "01SILREAR0VH00DS", "01SILFRONTVH00DS", "03SILREAR0VH00DS"
            else:
                root, dir_x, dir_y = "03SILREAR0VH00DS", "03SILFRONTVH00DS", "01SILREAR0VH00DS"
            occ_chs = [
                f"{pos}{loc}LOC0{self.__dummy_version[:2]}{self.__perc}DS"
                for pos in ("01", "03")
                for loc in ("HEAD", "CHST", "PELV")
            ]
            occ_chs = [ch for ch in occ_chs if ch in set(ch[:-2] for ch in db.columns)]

            # get coordinates as arrays
            occ_coords = np.array([db[[f"{occ_ch}{d}{cfc}" for d in "XYZ"]].to_numpy() for occ_ch in occ_chs])
            root_coord = db[[f"{root}{d}{cfc}" for d in "XYZ"]].to_numpy()
            dir_x_coord = db[[f"{dir_x}{d}{cfc}" for d in "XYZ"]].to_numpy()
            dir_y_coord = db[[f"{dir_y}{d}{cfc}" for d in "XYZ"]].to_numpy()
            z_axis = np.cross(dir_x_coord - root_coord, dir_y_coord - root_coord)
            z_axis /= np.linalg.norm(z_axis, axis=1)[:, np.newaxis]
            dir_z_coord = root_coord + z_axis

            # calculate
            local_disps: List[pd.DataFrame] = []
            new_cols = []
            for d, d_coord in zip("XYZ", (dir_x_coord, dir_y_coord, dir_z_coord)):
                ld = get_displ_along_axis(
                    nodes_coord=occ_coords, root_coord=root_coord, direction_coord=d_coord, as_displacement=True, from_root=True
                )
                LOG.debug("Local displacements along x axis have shape %s", ld.shape)
                nc = [f"{occ_ch[:6]}LOC0{self.__dummy_version[:2]}{self.__perc}DS{d}{cfc}" for occ_ch in occ_chs]
                new_cols.extend(nc)
                local_disps.append(pd.DataFrame(ld.T, index=db.index, columns=nc))

            # store
            db_new.drop(columns=new_cols, inplace=True)
            db_new = pd.concat([db_new, *local_disps], axis=1)

        LOG.info("Data with locals has shape %s", db_new.shape)
        return db_new


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
        "--dummy",
        required=True,
        type=str,
        help="Dummy type (e.g. TH2.1, TH2.7, H3Rigid)",
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
    reader = ReadFeModel(in_dir=args.directory, out_dir=args.out, dummy_v=args.dummy)
    db = reader.run()

    LOG.info("Data has shape %s", db.shape)
    LOG.debug("Data:\n%s", db)

    LOG.info("DONE")


if __name__ == "__main__":
    run()
