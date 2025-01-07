import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from DummyIds import DummyIds
from IsoMme import IsoMme
from lasso.dyna import Binout

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.ReadBinout as ReadBinout
from src.utils.CfCFilter import CfCFilter
from src.utils.local_displacement import get_displ_along_axis
from src.utils.PathChecker import PathChecker
from src.utils.UnifySignal import UnifySignal
from src.utils.UnitConverter import UnitConverter

LOG: logging.Logger = logging.getLogger(__name__)


class FeSimulation:
    def __init__(
        self,
        sim_dir: Path,
        tend_ms: float,
        sampling_rate_ms: float,
        unit_mass: Literal["g", "kg", "t"],
        unit_length: Literal["mm", "m"],
        unit_time: Literal["s", "ms"],
        mme: IsoMme,
        dids: DummyIds,
    ) -> None:
        """Read LS-Dyna simulation results from binout to selected channels with unified time axis
        and unit system [ms, mm, kg]

        Args:
            sim_dir (Path): directory containing binout file of unique simulation
            tend_ms (float): expected end time of simulation, signals will be extrapolated
            sampling_rate_ms (float): sampling rate  for signal interpolation
            unit_mass (Literal[&quot;g&quot;, &quot;kg&quot;, &quot;t&quot;]): unit of of mass
            unit_length (Literal[&quot;mm&quot;, &quot;m&quot;]): unit of length
            unit_time (Literal[&quot;s&quot;, &quot;ms&quot;]): unit of time
            mme (IsoMme): container with ISO MME naming conventions
            dids (DummyIds): container with dummy model specific IDs
            log (Union[logging.Logger, None], optional): log module. Defaults to None.
        """
        self.__checker = PathChecker()

        # check directory & binouts
        self.sim_dir = self.__checker.check_directory(path=sim_dir, exit=True)
        _ = self.__checker.check_file_type(path=self.sim_dir, file_pattern="binout*", exit=True)

        # init
        self.mme = mme
        self.dids = dids
        self.data_raw: Dict[str, pd.DataFrame] = {}
        self.data_unified: pd.DataFrame = pd.DataFrame()

        # expected max time
        self.tend_ms = tend_ms
        self.sampling_rate_ms = sampling_rate_ms
        self.tend_is = tend_ms

        # unit conversions
        self.__conv_facts = UnitConverter(unit_mass=unit_mass, unit_length=unit_length, unit_time=unit_time)

        # names
        self._binout_acc = "acceleration"
        self._binout_force = "force"
        self._binout_displacement = "displacement"
        self._binout_moment = (
            "beta_moment_damping"  # should be theta_moment_total but there seem to be a bug in Binout, confirmed with LS-PrePost
        )

    def fill_raw_db(
        self,
        binout: Binout,
        path: List[str],
        ids: Dict[str, str],
        conv: float,
    ):
        """Reads data for given sensor

        Args:
            binout (Binout): _description_
            path (List[str]): _description_
            ids (Dict[str, str]): _description_
            conv (float): _description_
        """
        LOG.debug("Read from binout path %s", path)

        # assemble channel names
        mapping = self.__assemble_channel_names(path=path, ids=ids)

        # read from binout
        fe: pd.DataFrame = binout.as_df(*path)

        # get relevant channels and rename
        existing_ids = list(set(ids.keys() & set(fe.columns)))
        if len(existing_ids) != len(ids):
            LOG.warning("Not all ids found in binout: %s", set(ids.keys()) - set(existing_ids))
        fe = fe[existing_ids].copy()
        fe.rename(columns=mapping, inplace=True)

        # convert to target unit system
        fe *= conv
        fe.index *= self.__conv_facts.time2ms()

        # store end time of simulation
        self.tend_is = fe.index.max() if fe.index.max() < self.tend_is else self.tend_is

        # store data
        LOG.debug("Add data with shape %s to raw database", fe.shape)
        if path[0] in self.data_raw:
            self.data_raw[path[0]] = pd.concat([self.data_raw[path[0]], fe], axis=1)
        else:
            self.data_raw[path[0]] = fe

    def __assemble_channel_names(self, path: List[str], ids: Dict[str, str]) -> Dict[str, str]:
        """map dummy ids to channel names

        Args:
            path (List[str]): path within binout file
            ids (Dict[str, str]): mapping of dummy ids to sensor location

        Returns:
            Dict[str, str]: mapping of dummy ids to channel names
        """
        # determine channel name parts
        if path[-1][1] == "_":
            # directional channels (e.g. x_acceleration, y_force)
            direction = path[-1][0].upper()
            dimension = {
                self._binout_acc: self.mme._acc,
                self._binout_force: self.mme._force,
            }[path[-1].split("_")[-1]]
        elif path[-1] == self._binout_displacement:
            # non directional displacement (e.g. displacement)
            direction, dimension = self.mme._directions[0], self.mme._displ
        elif path[-1] == self._binout_moment:
            # joint moment (e.g. beta_moment_damping)
            direction, dimension = self.mme._directions[1], self.mme._moment
        else:
            LOG.warning("MME incompatible binout path: %s", path)
            direction, dimension = "", ""

        # assign channel names to ids
        mapping = {}
        for key in ids.keys():
            # map dummy ids to channel names
            mapping[key] = self.mme.channel_name(sensor_loc=ids[key], dimension=dimension, direction=direction, cfc=None)

        return mapping

    def binout2raw(self):
        """Get sensor data from binout"""
        with ReadBinout.ReadBinout(sim_dir=self.sim_dir) as binout:
            # read directional data
            for dd in self.mme._directions:
                d = dd.lower()
                # nodal accelerations
                LOG.debug("Get nodal %s acceleration", dd)
                self.fill_raw_db(
                    binout=binout.binout,
                    path=["nodout", f"{d}_{self._binout_acc}"],
                    ids=self.dids._nodout_ids,
                    conv=self.__conv_facts.acceleration2g(),
                )

                # joint forces
                LOG.debug("Get joint %s forces", dd)
                self.fill_raw_db(
                    binout=binout.binout,
                    path=["jntforc", "joints", f"{d}_{self._binout_force}"],
                    ids=self.dids._jntforc_f_ids,
                    conv=self.__conv_facts.force2kn(),
                )

                # contact forces
                LOG.debug("Get contact %s forces", dd)
                try:
                    self.fill_raw_db(
                        binout=binout.binout,
                        path=["rcforc", f"{d}_{self._binout_force}"],
                        ids=self.dids._rcforc_ids,
                        conv=self.__conv_facts.force2kn(),
                    )
                except KeyError:
                    LOG.error("No contact forces found in binout")

            # chest potentiometer
            LOG.debug("Get chest potentiometer")
            self.fill_raw_db(
                binout=binout.binout,
                path=["deforc", self._binout_displacement],
                ids=self.dids._deforc_ids,
                conv=self.__conv_facts.chest_deflection(dummy=self.mme.dummy_type, percentile=self.mme.dummy_percentile),
            )

            # joint moments
            LOG.debug("Get joint moment")
            self.fill_raw_db(
                binout=binout.binout,
                path=["jntforc", "type1", self._binout_moment],
                ids=self.dids._jntforc_m_ids,
                conv=self.__conv_facts.moment2nm(),
            )

    def __unify_data(self) -> pd.DataFrame:
        """Interpolate raw data to given sampling rate, extrapolation via median
        Background: LS-Dyna can write different channels types with different sampling rates

        Returns:
            pd.DataFrame: interpolated data with shape (n time stamps new, n channels)
        """
        data_unified = []
        unifier = UnifySignal(target_tend_ms=self.tend_ms, target_sampling_rate_ms=self.sampling_rate_ms)

        # loop over channel types
        for src in self.data_raw.keys():
            LOG.debug("Process %s", src)
            data_unified.append(unifier.unify(db=self.data_raw[src]))

        # combine channels types
        data_unified = pd.concat(data_unified, axis=1)

        return data_unified

    def __apply_cfc_filter(self, db: np.ndarray, channels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply CFC filter to raw data

        Args:
            db (np.ndarray): data with shape (n time stamps, n channels)
            channels (List[str]): fixed channel names with shape (n channels,)

        Returns:
            Tuple[np.ndarray, List[str]]: filtered data with shape (n time stamps, n channels * n filter classes)
        """
        # init
        all_data, all_channels = [], []
        cfc_filter = CfCFilter()

        # loop over filter classes
        for cfc in self.mme.cfc.keys():
            # apply filter
            all_data.append(
                cfc_filter.filter(
                    tsp=self.sampling_rate_ms / self.__conv_facts.time2ms(),
                    signal=db,
                    cfc=cfc,
                )
            )

            # store filter class modified channel names
            all_channels.extend([f"{ch[:-1]}{self.mme.cfc[cfc]}" for ch in channels])

        return all_data, all_channels

    def __calculate_resultants(self):
        """Calculate resultants of directional data for each filter class"""
        # loop over filter classes
        for cfc in self.mme.cfc.keys():
            for ch_type in self.mme._resultants.keys():
                for location in self.mme._resultants[ch_type]:
                    # get channels names of directional data
                    ch_names = [self.mme.channel_name(location, ch_type, d, cfc) for d in self.mme._directions]
                    try:
                        # generate new channel name
                        r_name = self.mme.channel_name(location, ch_type, self.mme._res_direction, cfc)

                        # calculate resultant
                        res = np.linalg.norm(self.data_unified[ch_names].values, axis=1)

                        # store
                        self.data_unified[r_name] = res

                        # remove directional force channels
                        if ch_type == self.mme._force:
                            self.data_unified.drop(columns=ch_names, inplace=True)
                    except KeyError:
                        LOG.error("Channel names %s not found", ch_names)

    def raw2final(self):
        """Transform raw data to unified & CFC filtered data"""
        LOG.debug("Unify and CFC filter data")

        # unify data
        data_unified = self.__unify_data()

        # filter data
        channel_collector = sorted(data_unified.columns)
        all_data, all_channels = self.__apply_cfc_filter(db=data_unified[channel_collector].values, channels=channel_collector)

        # concat
        self.data_unified = pd.DataFrame(np.concatenate(all_data, axis=1), index=data_unified.index, columns=all_channels)
        self.data_unified.index.name = self.mme._time

        # add resultants
        self.__calculate_resultants()

    def add_local_displacements(self):
        # define
        chs = {
            82000028: "01SILFRONTVH00DS",
            82000012: "01SILREAR0VH00DS",
            82000020: "03SILFRONTVH00DS",
            82000004: "03SILREAR0VH00DS",
            68000001: self.mme.channel_name(sensor_loc="HEAD", dimension="DS", direction="X", cfc="D")[:-2],
            68001787: self.mme.channel_name(sensor_loc="CHST", dimension="DS", direction="X", cfc="D")[:-2],
            68003304: self.mme.channel_name(sensor_loc="PELV", dimension="DS", direction="X", cfc="D")[:-2],
        }
        rel_ids = [str(x) for x in chs.keys()]

        # get data
        db = []
        with ReadBinout.ReadBinout(sim_dir=self.sim_dir) as binout:
            for d in "xyz":
                data = binout.binout.as_df("nodout", f"{d}_coordinate")
                ids = list(set(rel_ids) & set(data.columns))
                if len(ids) != len(rel_ids):
                    LOG.warning("Not all ids found in binout: %s", set(rel_ids) - set(ids))
                data = data[ids].copy()
                data = data.rename(columns={str(k): v + d.upper() for k, v in chs.items()})
                db.append(data)
        db = pd.concat(db, axis=1)
        db.index *= self.__conv_facts.time2ms()
        db *= self.__conv_facts.length2mm()

        # unify
        unifier = UnifySignal(target_tend_ms=self.tend_ms, target_sampling_rate_ms=self.sampling_rate_ms)
        db = unifier.unify(db=db)

        # filter
        cfc_filter = CfCFilter()
        filts = []
        for cfc in self.mme.cfc.keys():
            # apply filter
            filt = cfc_filter.filter(
                tsp=self.sampling_rate_ms / self.__conv_facts.time2ms(),
                signal=db.values,
                cfc=cfc,
            )
            filts.append(pd.DataFrame(filt, index=db.index, columns=[f"{ch}{self.mme.cfc[cfc]}" for ch in db.columns]))
        db = pd.concat(filts, axis=1)
        del filts
        del filt

        # calculate
        local_disps: List[pd.DataFrame] = []
        for cfc in self.mme.cfc.values():
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
                self.mme.channel_name(sensor_loc=loc, dimension="DS", direction="X", cfc="D")[:-2]
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
            for d, d_coord in zip("XYZ", (dir_x_coord, dir_y_coord, dir_z_coord)):
                ld = get_displ_along_axis(
                    nodes_coord=occ_coords, root_coord=root_coord, direction_coord=d_coord, as_displacement=True, from_root=True
                )
                LOG.debug("Local displacements along x axis have shape %s", ld.shape)

                local_disps.append(
                    pd.DataFrame(
                        ld.T,
                        index=db.index,
                        columns=[
                            self.mme.channel_name(sensor_loc=f"{occ_ch[2:6]}LOC", dimension="DS", direction=d, cfc=cfc)
                            for occ_ch in occ_chs
                        ],
                    )
                )
                local_disps[-1].index.name = self.mme._time
        db = pd.concat(local_disps, axis=1)

        # store
        self.data_unified = pd.concat([self.data_unified, db], axis=1)
        LOG.info("Added local displacements of shape %s", db.shape)

    def save(self):
        """Store unified data"""
        out_path = self.sim_dir / "channels.parquet"
        LOG.info("Store unified data to %s", out_path)
        self.data_unified.to_parquet(self.sim_dir / "channels.parquet", index=True)


def test():
    mme = IsoMme(dummy_percentile=50, dummy_position="03", dummy_type="H3")
    d_path = Path("data")

    with tempfile.TemporaryDirectory() as tmp:
        LOG.info("Working directory is %s", tmp)
        zip_file = d_path / "samples" / "binouts.zip"
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(tmp)

        simulation = FeSimulation(
            tend_ms=140,
            sampling_rate_ms=0.1,
            unit_mass="t",
            unit_length="mm",
            unit_time="s",
            sim_dir=Path(tmp),
            mme=mme,
            dids=DummyIds(mme=mme),
        )

        simulation.binout2raw()
        simulation.raw2final()
        simulation.add_local_displacements()
        print(simulation.data_unified)
        for ch in sorted(simulation.data_unified.columns):
            print(ch)

    LOG.info("DONE")


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
