import argparse
import datetime
import logging
import multiprocessing
import sys
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.data.fe_processing.InjuryCalculator import InjuryCalculator
from src.data.fe_processing.IsoMme import IsoMme
from src.utils.hash_file import hash_file

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def main():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        type=Path,
        help="Path to directory data",
    )
    parser.add_argument(
        "-c",
        "--n_cpu",
        required=False,
        default=1,
        type=int,
        help="Number of parallel processes (default is %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set log level to DEBUG")

    # parse
    args = parser.parse_args()

    # init logger
    custom_log.init_logger(log_lvl=logging.DEBUG if args.verbose else logging.INFO)
    LOG.info("START")

    # set number of used parallel processes
    n_cpu = args.n_cpu if args.n_cpu <= multiprocessing.cpu_count() else multiprocessing.cpu_count()

    # check data
    if args.directory.is_dir():
        LOG.info("Directory %s exists", args.directory)
    else:
        LOG.critical("Directory does not exist - EXIT", args.directory)
        sys.exit(1)

    f_path_channels, f_path_results_info, f_path_injury = [
        args.directory / f_name for f_name in (STR.fname_channels, STR.fname_results_info, STR.fname_injury_crit)
    ]
    for file in (f_path_channels, f_path_results_info):
        if file.is_file():
            LOG.info("File %s exists", file)
        else:
            LOG.critical("File does not exist - EXIT", file)
            sys.exit(1)
    if f_path_injury.is_file():
        LOG.warning("File %s exists - Overwrite", f_path_injury)
    else:
        LOG.info("File %s does not exist - New one will be generated", f_path_injury)

    # get hash reference
    info_file = json_util.load(f_path=f_path_results_info)
    hash_reference = info_file[STR.output]["Channels"][STR.hash]
    LOG.info("Got hash %s for %s", hash_reference, f_path_channels)

    # check hash
    hash_current = hash_file(fpath=f_path_channels)
    if hash_current == hash_reference:
        LOG.info("Hashes match - continue")
    else:
        LOG.critical("Hashes do not match - EXIT")
        sys.exit(1)

    # get sims
    LOG.info("Get sim ids from %s", f_path_channels)
    sim_ids = (
        pd.read_parquet(f_path_channels, columns=[STR.id, STR.perc], filters=[(STR.time, "==", 0)]).droplevel(STR.time).index
    )
    LOG.info("Got %s sim ids with level %s", sim_ids.shape, sim_ids.names)

    # calculate
    LOG.info("Start calculation for %s sims and %s processes", sim_ids.shape, n_cpu)
    with multiprocessing.Pool(processes=n_cpu, maxtasksperchild=1) as pool:
        injury_crit = pool.starmap(func=calculate, iterable=product(sim_ids, [sim_ids.names], [f_path_channels]))
    LOG.info("Parallel processing done - got %s elements", len(injury_crit))

    # join
    injury_crit = pd.concat(injury_crit, copy=False)
    LOG.info("Got injury_crit %s", injury_crit.shape)

    # store
    LOG.info("Overwrite %s with injury_crit", f_path_injury)
    injury_crit.to_parquet(f_path_injury, index=True)

    # document
    LOG.info("Document results in %s", f_path_results_info)
    hash_new = hash_file(fpath=f_path_injury)
    info_file[STR.creation] = str(datetime.datetime.now())
    info_file[STR.output]["Injury Criteria"][STR.hash] = hash_new
    info_file[STR.output]["Injury Criteria"][STR.path] = f_path_injury
    json_util.dump(f_path=f_path_results_info, obj=info_file)

    LOG.info("DONE")


def calculate(sim_idx: tuple, idx_lvls: List[str], f_path_channels: Path) -> pd.DataFrame:
    # get data
    LOG.debug("Get data for sim %s - read %s", sim_idx, f_path_channels)
    db = pd.read_parquet(
        f_path_channels,
        filters=[(idx_lvls[0], "==", sim_idx[0]), (idx_lvls[1], "==", sim_idx[1])],
        columns=[
            "03HEAD0000OCCUACRD",
            "03CHST0000OCCUDSXD",
            "03CHST0000OCCUACRD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03FEMRRI00OCCUFOZD",
            "03FEMRLE00OCCUFOZD",
        ],
    ).droplevel([STR.id, STR.perc])
    perc = int(sim_idx[1])
    db.rename(columns={c: c.replace("OCCU", "H3" + f"{perc:02d}") for c in db.columns}, inplace=True)

    # calculate
    LOG.debug("Calculate injury for sim %s", sim_idx)
    inj = InjuryCalculator(data=db, mme=IsoMme(dummy_type="H3", dummy_percentile=perc, dummy_position="03"), cfc="D")
    inj.calculate()

    # format
    injury_crit = pd.DataFrame(inj.injury_crit, index=pd.MultiIndex.from_tuples([sim_idx], names=idx_lvls))
    LOG.debug("Got injury_crit %s", injury_crit.shape)

    return injury_crit


if __name__ == "__main__":
    main()
