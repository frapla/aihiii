import argparse
import datetime
import logging
import multiprocessing as mp
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from DummyIds import DummyIds
from FeSimulation import FeSimulation
from InjuryCalculator import InjuryCalculator
from IsoMme import IsoMme
from SimProperty import SimProperty

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.NameSpace2Json as book
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


class ProcessSimulation:
    def __init__(self, sim_dir: Path, sim_prop: Union[SimProperty, None] = None) -> None:
        """Extract data from binout files of single simulation

        Args:
            sim_dir (Path): directory with simulation data
            sim_prop (Union[SimProperty, None], optional): _description_. Defaults to None.
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # init simulation's properties
        self.sim_prop: SimProperty = SimProperty() if sim_prop is None else sim_prop

        # set working directory
        self.sim_dir = PathChecker().check_directory(path=sim_dir, exit=True)
        self.work_dir = self.sim_dir
        self.signal_fname = "channels.parquet"
        self.injury_fname = "injury_criteria.parquet"
        self.book_fname = "sim_info"

        # book keeping
        self._date = str(datetime.datetime.now())
        self._sim_dir = self.sim_dir.absolute()
        self._input_files = []
        self._created_signals_file = Path()
        self._created_injury_path = Path()
        self._dummy_percentile = self.sim_prop.dummy_percentile
        self._dummy_position = self.sim_prop.dummy_position
        self._dummy_type = self.sim_prop.dummy_type
        self._set_end_time = self.sim_prop.tend_ms
        self._set_sampling_rate = self.sim_prop.sampling_rate_ms
        self._input_unit_system = self.sim_prop.unit_system_in
        self._set_unit_system = ("kg", "mm", "ms")
        self._input_end_time = 0

        # set context
        self.mme = IsoMme(
            dummy_percentile=self._dummy_percentile,
            dummy_position=self._dummy_position,
            dummy_type=self._dummy_type,
        )
        self.dummy_ids = DummyIds(mme=self.mme)

        # inits
        self.signals = pd.DataFrame()
        self.injury_criteria = pd.DataFrame()

    def process(self):
        """Process simulation"""
        # temp folder
        proc = mp.current_process()
        with tempfile.TemporaryDirectory(prefix=f"{proc.pid}_{proc.name}_", suffix=__name__) as temp_dir:
            LOG.debug("Work in temp directory %s", temp_dir)
            self.work_dir = Path(temp_dir)

            # get data
            LOG.debug("Get data from binouts")
            self.extract_signal_data()
            self.calculate_injury_criteria()

            # store extracted channels
            LOG.debug("Store extracted data")
            self.store_data()

    def store_data(self):
        """Store extracted data to zip archive"""
        # store extracted channels
        LOG.debug("Store extracted channels to %s", self.signal_fname)
        out_path = self.sim_dir / self.signal_fname
        self.signals.to_parquet(out_path, index=True)
        self._created_signals_file = out_path

        # store calculated injury values
        LOG.debug("Store calculated injury values to %s", self.injury_fname)
        out_path = self.sim_dir / self.injury_fname
        self.injury_criteria.to_parquet(out_path, index=True)
        self._created_injury_path = out_path

        # store bookkeeping
        LOG.debug("Store simulation information to %s", self.book_fname)
        book_keeper = book.NameSpace2Json()
        book_keeper.get_attr(self)
        book_keeper.to_json(f_path=self.sim_dir / self.book_fname)

    def extract_signal_data(self):
        """Extract signals from binout"""
        # treat zip archive
        binout_archives = list(self.sim_dir.glob("binout*.zip"))
        if len(binout_archives) == 1:
            binout_archive = binout_archives[0]
            self._input_files = [binout_archive.name]
            LOG.debug("Extract data to %s", self.work_dir)
            with zipfile.ZipFile(binout_archive, "r") as zip_ref:
                zip_ref.extractall(self.work_dir)
        else:
            LOG.debug("Copy binout files to %s", self.work_dir)
            for binout_file in self.sim_dir.glob("binout*"):
                self._input_files.append(binout_file.name)
                shutil.copy(src=binout_file, dst=self.work_dir)

        # read simulation
        LOG.debug("Extract signals from binout")
        simulation = FeSimulation(
            tend_ms=self._set_end_time,
            sampling_rate_ms=self._set_sampling_rate,
            unit_mass=self._input_unit_system[0],
            unit_length=self._input_unit_system[1],
            unit_time=self._input_unit_system[2],
            sim_dir=self.work_dir,
            mme=self.mme,
            dids=self.dummy_ids,
        )
        simulation.binout2raw()
        self._input_end_time = simulation.tend_is
        simulation.raw2final()
        simulation.add_local_displacements()

        # store
        self.signals = simulation.data_unified.copy()
        del simulation

    def calculate_injury_criteria(self):
        """Calculate injury criteria from filtered signals"""
        injury_criteria = {}

        # calculate for multiple filter classes
        for cfc in self.mme.cfc.values():
            LOG.debug("Calculate injury for CFC %s", cfc)
            calculator = InjuryCalculator(data=self.signals, mme=self.mme, cfc=cfc)
            calculator.calculate()
            injury_criteria[cfc] = pd.Series(calculator.injury_crit)

        # convert
        injury_criteria = pd.DataFrame(injury_criteria).T
        injury_criteria.index.name = "CFC"

        # store
        self.injury_criteria = injury_criteria


def cmd_line() -> Tuple[Path, SimProperty]:
    sim_prop: SimProperty = SimProperty()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        required=True,
        help="Directory with binout files",
    )
    parser.add_argument(
        "--dummy_percentile",
        "-p",
        type=int,
        required=False,
        default=sim_prop.dummy_percentile,
        help="Dummy percentile (Default %(default)s)",
    )
    parser.add_argument(
        "--dummy_position",
        "-s",
        type=str,
        required=False,
        default=sim_prop.dummy_position,
        help="Dummy position (Default %(default)s)",
    )
    parser.add_argument(
        "--dummy_type",
        "-t",
        type=str,
        required=False,
        default=sim_prop.dummy_type,
        help="Dummy type (Default %(default)s)",
    )
    parser.add_argument(
        "--tend_ms",
        "-e",
        type=float,
        required=False,
        default=sim_prop.tend_ms,
        help="Expected end time of simulation (Default %(default)s)",
    )
    parser.add_argument(
        "--sr_ms",
        "-r",
        type=float,
        required=False,
        default=sim_prop.sampling_rate_ms,
        help="Expected sampling rate of signals (Default %(default)s)",
    )
    parser.add_argument(
        "--units",
        "-u",
        type=tuple,
        required=False,
        default=sim_prop.unit_system_in,
        help="Input unit system (Default %(default)s)",
    )
    args = parser.parse_args()

    return args.directory, SimProperty(
        perc=args.dummy_percentile,
        pos=args.dummy_position,
        dbuild=args.dummy_type,
        tend_ms=args.tend_ms,
        sr_ms=args.sr_ms,
        units=args.units,
    )


def test():
    sim_dir, sim_prop = cmd_line()

    if sim_dir.is_dir():
        LOG.info("Process simulation in %s", sim_dir)
        proc = ProcessSimulation(sim_dir=sim_dir, sim_prop=sim_prop)
        proc.process()
        LOG.info("DONE")
    else:
        LOG.error("Directory %s not found", sim_dir)


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.DEBUG)
    test()
