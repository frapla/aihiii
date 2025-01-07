import argparse
import datetime
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from ProcessSimulation import ProcessSimulation
from SimProperty import SimProperty
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


def process_simulation(sim_dir: Path, sim_prop: SimProperty, log_lvl: int, override: bool):
    # start logger
    LOG.warning("Start processing directory %s", sim_dir)

    proc = ProcessSimulation(
        sim_dir=sim_dir,
        sim_prop=sim_prop,
    )

    if override or not (sim_dir / "channels.parquet").is_file() or not (sim_dir / "injury_criteria.parquet").is_file():
        proc.process()
    LOG.warning("Done processing directory %s", sim_dir)

    return int(sim_dir.stem[1:]), proc._input_end_time / sim_prop.tend_ms


class ProcessDirectory:
    def __init__(self, simulations_dpath: Path, doe_fpath: Path, unite_only: bool, cfc: str, override: bool) -> None:
        # init log
        self.__checker = PathChecker()
        self.str = StandardNames()
        self.unite_only = unite_only
        self.cfc = cfc
        self.override = override

        # init doe directory
        self.simulation_dpath = self.__checker.check_directory(path=simulations_dpath, exit=True)

        # get doe xlsx
        self.doe_fpath = self.__checker.check_file(path=doe_fpath, exit=True)
        self.doe = pd.read_parquet(doe_fpath)
        self.doe.index.name = self.str.sim_id
        self.sim_ids = set(f"V{sid:07d}" for sid in self.doe.index)
        self.doe_renamer = self.__doe_single_idx()

        # init bookkeeping
        self._date = str(datetime.datetime.now())
        self.book_fname = "results_info"
        self._doe_dir = self.simulation_dpath.absolute()
        self._doe_xlsx = doe_fpath.absolute()
        self._channel_fpath = (self.doe_fpath.parent / "channels.parquet").absolute()
        self._injury_fpath = (self.doe_fpath.parent / "injury_criteria.parquet").absolute()
        self._renamer_fpath = (self.doe_fpath.parent / "sim_id_2_id.parquet").absolute()
        self._n_simulations = 0
        self._dummy_position = "03"
        self._dummy_type = "H3"
        self._tend_ms = 140
        self._sampling_rate_ms = 0.1
        self._unit_system_in = ("kg", "mm", "ms")
        self._end_time_bigger_than = {0.9: [], 0.8: [], 0.7: [], 0: []}
        self._hashed_channels = {}
        self._hashed_injury = {}

    def run(self, n_cpu: int):
        t_start = time.time()
        # get simulations
        LOG.info("Start processing simulations")
        in_tuples = self.__scan_dir()
        LOG.debug("Example tuple for multiprocessing: %s", in_tuples[0])

        # start
        if self.unite_only:
            LOG.warning("Only unite data - no processing")
        else:
            LOG.info("Process %s directories", self._n_simulations)
            with multiprocessing.Pool(processes=n_cpu) as pool:
                t_end_perc = pool.starmap(
                    func=process_simulation,
                    iterable=in_tuples,
                    chunksize=int(self._n_simulations / n_cpu),
                )
            self.__eval_end_times(t_end_perc=t_end_perc)
        # collect and store all data
        self.doe_renamer.to_parquet(self._renamer_fpath, index=True)
        self._hashed_channels = self.__concat_and_store_data(out_fpath=self._channel_fpath)
        self._hashed_injury = self.__concat_and_store_data(out_fpath=self._injury_fpath)

        # book keeping

        LOG.info("Store directory information to %s", self.book_fname)
        book = {
            self.str.creation: self._date,
            self.str.input: {
                self.str.path: self.doe_fpath,
                self.str.hash: hash_file(self.doe_fpath),
            },
            self.str.output: {
                "Channels": {
                    self.str.path: self._channel_fpath,
                    self.str.hash: hash_file(self._channel_fpath),
                },
                "Injury Criteria": {
                    self.str.path: self._injury_fpath,
                    self.str.hash: hash_file(self._injury_fpath),
                },
                "Renamer": {
                    self.str.path: self._renamer_fpath,
                    self.str.hash: hash_file(self._renamer_fpath),
                },
            },
            "Infos": {
                "n Simulations": self._n_simulations,
                "Target End Time": self._tend_ms,
                "Sampling Rate": self._sampling_rate_ms,
                "Unit System": self._unit_system_in,
                "Dummy Position": self._dummy_position,
                "Dummy Type": self._dummy_type,
                "CFC": self.cfc,
                "End Time": self._end_time_bigger_than,
            },
            "Data": {
                "Channels": self._hashed_channels,
                "Injury": self._hashed_injury,
            },
        }
        book_path = self.doe_fpath.parent / self.book_fname
        LOG.info("Stored book in %s", book_path)
        json_util.dump(f_path=book_path, obj=book)

        # done
        LOG.info(
            "%s Simulation processed - runtime %s",
            self._n_simulations,
            time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start)),
        )

    def __doe_single_idx(self) -> pd.DataFrame:
        doe = self.doe.copy().apply(pd.to_numeric, downcast="float")
        doe = doe.reset_index().set_index(sorted(set(doe.columns) - {self.str.perc}))
        uniques = doe.index.drop_duplicates()
        renamer = dict(zip(uniques, range(len(uniques))))
        doe.index = [renamer[idx] for idx in doe.index]
        doe.index.name = self.str.id
        doe = doe.reset_index().set_index(self.str.sim_id)

        return doe

    def __concat_and_store_data(self, out_fpath: Path):
        LOG.info("Concatenate and store data for %s", out_fpath.stem)
        dbs, hashed_channels = [], {}
        for sim_id in tqdm(sorted(self.sim_ids)):
            LOG.debug("Read Data for %s", sim_id)
            fpath = self.simulation_dpath / sim_id / out_fpath.name
            if fpath.is_file():
                hashed_channels[sim_id] = {self.str.path: fpath, self.str.hash: hash_file(fpath)}
                db = pd.read_parquet(fpath).apply(pd.to_numeric, downcast="float")
                if out_fpath.stem == "channels":
                    db = db[sorted([ch for ch in db.columns if ch.endswith(self.cfc)])].copy()
                    db.rename(columns={ch: ch[:10] + "OCCU" + ch[-4:] for ch in db.columns}, inplace=True)
                else:
                    db = pd.DataFrame(db.loc[self.cfc]).T.copy()
                idxs = self.doe_renamer.loc[int(sim_id[1:])]
                db[self.str.id] = idxs[self.str.id]
                db[self.str.perc] = idxs[self.str.perc]
                db.set_index([self.str.id, self.str.perc], append=True if out_fpath.stem == "channels" else False, inplace=True)
                LOG.debug("Got data of shape %s", db.shape)
                dbs.append(db)
        del db
        LOG.info("Concatenate %s dataframes", len(dbs))
        dbs = pd.concat(dbs, axis=0, copy=False)
        LOG.info("Got data of shape %s", dbs.shape)

        LOG.info("Store to %s", out_fpath)
        dbs.to_parquet(out_fpath, index=True)

        return hashed_channels

    def __eval_end_times(self, t_end_perc):
        end_perc_lvls = sorted(self._end_time_bigger_than.keys(), reverse=True)
        for sim_id, end_perc in t_end_perc:
            if end_perc < end_perc_lvls[-2]:
                self._end_time_bigger_than[end_perc_lvls[-1]].append(sim_id)
            else:
                for lvl in end_perc_lvls:
                    if end_perc >= lvl:
                        self._end_time_bigger_than[lvl].append(sim_id)
                        break

    def __scan_dir(self) -> List[Tuple[Path, SimProperty, int]]:
        LOG.info("Scan directory for valid simulation directories")
        sim_dirs = []
        for pot_dir in self.simulation_dpath.glob("*"):
            if pot_dir.is_dir() and list(pot_dir.glob("binout*")) and pot_dir.stem in self.sim_ids:
                sim_dirs.append(pot_dir)
        self._n_simulations = len(sim_dirs)
        LOG.debug("Count of valid simulation directories is %s", self._n_simulations)

        # assemble sub process tuples
        LOG.debug("Assemble input for parallel processing")
        in_tuples = [
            (
                sim_dir,
                SimProperty(perc=self.doe.loc[int(sim_dir.stem[1:]), "PERC"]),
                LOG.level,
                self.override,
            )
            for sim_dir in sim_dirs
        ]

        return in_tuples


def from_cmd_line():
    # evaluate command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation_dpath",
        required=True,
        type=Path,
        help="Path to directory with simulation directories",
    )
    parser.add_argument(
        "--doe_fpath",
        required=True,
        type=Path,
        help="Path to parquet file describing the DOE",
    )
    parser.add_argument(
        "--n_cpu",
        required=False,
        default=1,
        type=int,
        help="Number of parallel processes (default is %(default)s)",
    )
    parser.add_argument(
        "--filter_class",
        "-f",
        required=False,
        default="D",
        type=str,
        help="CFC Filter Class to store in main file (default is %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set log level to DEBUG")
    parser.add_argument("-u", "--unite_only", action="store_true", help="Only unite data - no processing")
    parser.add_argument("-o", "--override", action="store_true", help="Read all new and overwrite existing files")
    args = parser.parse_args()

    # init logger
    custom_log.init_logger(log_lvl=logging.DEBUG if args.verbose else logging.INFO)

    # set number of used parallel processes
    n_cpu = args.n_cpu if args.n_cpu <= multiprocessing.cpu_count() else multiprocessing.cpu_count()

    # run
    if args.simulation_dpath.is_dir() and args.doe_fpath.is_file():
        LOG.info("Input directory: %s", args.simulation_dpath)
        proc = ProcessDirectory(
            simulations_dpath=args.simulation_dpath,
            doe_fpath=args.doe_fpath,
            unite_only=args.unite_only,
            cfc=args.filter_class,
            override=args.override,
        )
        proc.run(n_cpu=n_cpu)
    else:
        LOG.critical(
            "Input directory %s or doe xlsx %s does not exist - EXIT",
            args.simulation_dpath,
            args.doe_fpath,
        )
        sys.exit()


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    multiprocessing.freeze_support()
    from_cmd_line()
