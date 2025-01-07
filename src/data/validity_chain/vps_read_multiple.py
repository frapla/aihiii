import argparse
import logging
import shutil
import sys
from pathlib import Path
import multiprocessing as mp

import pandas as pd

sys.path.append(str(Path(__file__).absolute().parents[3]))
import warnings

import src.utils.custom_log as custom_log
from src.data.validity_chain.ReadVPSModel import ReadVPSModel

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

LOG: logging.Logger = logging.getLogger(__name__)


def run_parallel(file):
    LOG.info("Reading file %s", file)

    if file.stem.startswith("VH"):
        # for file naming e.g. VH_AM50_Config0001_THI_RESULT.erfh5 and VH_AF05_Config0001_THI_RESULT.erfh5
        dummy = file.stem.split("_")[1]
        dummy_v = "VI"
        dummy_perc = int(dummy[-2:])
    else:
        # for file naming e.g. FullFrontal_AM50_HIII_RESULT.erfh5
        dummy = file.stem.split("_")[1]
        dummy_v = {"AM": "H3", "VH": "VI"}[dummy[:2]]
        dummy_perc = int(dummy[-2:])

    reader = ReadVPSModel(
        in_dir=file.parent,
        out_dir=file.parent,
        dummy_v=dummy_v,
        dummy_perc=dummy_perc,
    )
    db = reader.run()

    LOG.info("Data has shape %s", db.shape)
    LOG.debug("Data:\n%s", db)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        help="Path to directory with single erfh5 file",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=logging.INFO,
        type=int,
        help="Log level (default is %(default)s)",
    )
    parser.add_argument(
        "--n_cpu",
        required=False,
        default=mp.cpu_count(),
        type=int,
        help="Log level (default is %(default)s)",
    )
    args = parser.parse_args()

    # init logger
    custom_log.init_logger(log_lvl=args.log_lvl)

    # check directory
    if args.directory.is_dir():
        LOG.info("Directory exists %s", args.directory)
        in_dir: Path = args.directory
    else:
        raise FileNotFoundError(f"Directory does not exist {args.directory}")

    # get files
    files = list(in_dir.glob("*.erfh5"))
    LOG.debug("Files found: %s", files)
    if len(files) > 1:
        LOG.warning("Multiple files found - move them in directories")
        read_files = []
        for file in files:
            new_dir = in_dir / file.stem
            LOG.info("Moving file %s to %s", file, new_dir)
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(src=file, dst=new_dir)
            read_files.append(new_dir / file.name)
    elif len(files) == 1:
        read_files = files
    else:
        read_files = list(in_dir.rglob("*.erfh5"))
        if read_files:
            LOG.info("Files found in subdirectories: %s", read_files)
        else:
            raise FileNotFoundError(f"No erfh5 files found in {in_dir}")

    # read files
    if args.n_cpu == 1:
        LOG.info("Running sequentially")
        map(run_parallel, read_files)
    else:
        LOG.info("Running in parallel with %s CPUs", args.n_cpu)
        with mp.Pool(processes=args.n_cpu) as pool:
            pool.map(run_parallel, read_files)

    LOG.info("DONE")


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    run()
