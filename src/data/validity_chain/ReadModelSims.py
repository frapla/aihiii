import argparse
import datetime
import logging
import multiprocessing as mp
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
import src.utils.parallel_handler as p_handler
from src.data.validity_chain.ReadFeModel import ReadFeModel
from src.utils.Csv import Csv
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


def read_results(sim_dir: Path, dummy_version: str, read_new: bool) -> pd.DataFrame:
    # start
    LOG.debug("Spawn process for %s", sim_dir)
    try:
        checker = PathChecker()

        reader = ReadFeModel(in_dir=sim_dir, out_dir=sim_dir, dummy_v=dummy_version)
        existing_results = checker.check_file_type(path=sim_dir, file_pattern=f"{reader.out_file_stem}*", exit=False)

        if existing_results and not read_new:
            LOG.info("Simulation %s already evaluated", sim_dir)
            csv = Csv(csv_path=sim_dir / reader.out_file_stem, compress=True)
            sim_data = csv.read()
        else:
            LOG.info("Process simulation in %s", sim_dir)
            sim_data = reader.run()

        LOG.info("Got data with shape %s for %s", sim_data.shape, sim_dir)

        # reformat
        data = defaultdict(list)
        for channel in sim_data.columns:
            data["Time"].extend(sim_data.index)
            data["Value"].extend(sim_data[channel])
            data["Channel"].extend([channel] * sim_data.shape[0])
            data["Source"].extend(["CAE THI"] * sim_data.shape[0])
            data["Assembly"].extend([sim_dir.parent.stem] * sim_data.shape[0])
            data["Configuration"].extend([sim_dir.parent.parent.stem] * sim_data.shape[0])
            case = sim_dir.stem
            data["Case"].extend([reader.standard_cases[case] if case in reader.standard_cases else case] * sim_data.shape[0])
            data["Side"].extend([case[-2:] if case[-3] == "_" else pd.NA] * sim_data.shape[0])

        return pd.DataFrame(data)
    except Exception:
        raise p_handler.ParallelException("".join(traceback.format_exception(*sys.exc_info())), mp.current_process().name)


class ReadModelSims:
    def __init__(self, root_dir: Path, dummy_version: str, read_new: bool = True) -> None:
        self.__checker = PathChecker()

        # check dir
        self.__root_dir = self.__checker.check_directory(path=root_dir, exit=True)

        # get sub directories
        self.__sim_dirs = [p for p in self.__root_dir.glob("**/**") if list(p.glob("binout*"))]
        LOG.info("Found %s directories with simulation data", len(self.__sim_dirs))
        self.__read_new = read_new

        # set dummy version
        self.dummy_version = dummy_version

    def read_data(self) -> pd.DataFrame:
        for_parallel = [(sim_dir, self.dummy_version, self.__read_new) for sim_dir in self.__sim_dirs]

        n_cpu = min(mp.cpu_count(), len(for_parallel))
        with mp.Pool(processes=n_cpu) as pool:
            workers = pool.starmap_async(func=read_results, iterable=for_parallel, error_callback=p_handler.handler)
            pool.close()
            pool.join()
            if workers.successful():
                sim_data = workers.get()
            else:
                sim_data = []

        sim_data = pd.concat(sim_data, ignore_index=True)

        LOG.info("Got data in shape %s", sim_data.shape)
        return sim_data

    def store(self, sim_data: pd.DataFrame):
        # write csv
        LOG.info("Store Data of shape %s", sim_data.shape)
        csv = Csv(csv_path=self.__root_dir / "extracted", compress=True)
        csv_path = csv.write(sim_data)
        csv_hash = hash_file(fpath=csv_path)

        # get sub info
        data_infos = {str(fp.absolute()): json_util.load(f_path=fp / "data_info") for fp in self.__sim_dirs}

        # store info
        json_util.dump(
            obj={
                "Creation": str(datetime.datetime.now()),
                "Input": data_infos,
                "Output": {
                    "Directory": self.__root_dir.absolute(),
                    "Database": csv_path.name,
                    "Data Hash": csv_hash,
                },
            },
            f_path=self.__root_dir / "data_info",
        )


def eval_cmd() -> argparse.Namespace:
    """Evaluate command line

    Returns:
        argparse.Namespace: command line arguments
    """
    # init
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        help="Path to directory with assemblies files",
    )
    parser.add_argument(
        "--dummy",
        required=True,
        type=str,
        help="Dummy type (e.g. TH2.1, TH2.7, H3Rigid)",
    )
    parser.add_argument(
        "--read_new",
        required=False,
        default=True,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Read data fresh from scratch if True (default is %(default)s)",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=logging.INFO,
        type=int,
        help="Log level (default is %(default)s)",
    )

    return parser.parse_args()


def test() -> None:
    # command line
    args = eval_cmd()
    custom_log.init_logger(log_lvl=args.log_lvl)

    # log
    LOG.debug("Command line input: %s", args)

    # run
    reader = ReadModelSims(root_dir=args.directory, dummy_version=args.dummy, read_new=args.read_new)
    data = reader.read_data()
    reader.store(sim_data=data)
    LOG.info("DONE")


if __name__ == "__main__":
    mp.freeze_support()
    test()
