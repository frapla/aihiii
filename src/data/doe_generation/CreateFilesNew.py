import argparse
import sys
import logging
from pathlib import Path
from typing import List

import pandas as pd
import tqdm

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
from src.data.doe_generation.KeyFileGenerator import create_key_file

LOG: logging.Logger = logging.getLogger(__name__)


def create_files(b_path: Path, doe_name: str = "doe.parquet", sim_name: str = "simulations"):
    """Generate key files from DOE

    Args:
        b_path (Path): directory path
        doe_name (str, optional): name of DOE file. Defaults to "doe.parquet".
        sim_name (str, optional): name of simulation directory. Defaults to "simulations".
    """
    # read doe
    doe = pd.read_parquet(b_path / doe_name)

    # create cases
    case_paths: List[Path] = []

    # create directory
    sim_dir = b_path / sim_name
    sim_dir.mkdir()

    # generate cases
    for idx in tqdm.tqdm(doe.index):
        body = create_key_file(
            rid=idx,
            percentile=doe.loc[idx, "PERC"],
            pab_t_vent=doe.loc[idx, "PAB_Vent_T"],
            pab_m_scal=doe.loc[idx, "PAB_M_Scal"],
            sll=doe.loc[idx, "SLL"],
            pulse_scale=doe.loc[idx, "Pulse_X_Scale"],
            pulse_angle_deg=doe.loc[idx, "Pulse_Angle"],
        )
        case_name = body[2]
        run_dir = sim_dir / case_name
        LOG.debug("Create %s", run_dir)
        run_dir.mkdir(exist_ok=True)
        case_paths.append(run_dir / f"{case_name}.key")
        with open(case_paths[-1], "w") as f:
            f.writelines([line + "\n" for line in body])

    LOG.info("Done")


def main():
    """run"""
    # init
    parser = argparse.ArgumentParser(description="Generate DOE by SOBOL sequence")

    # arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Directory to fetch data from",
        required=True,
    )
    parser.add_argument(
        "--doe_name",
        default="doe.parquet",
        help="DOE file name (default: %(default)s)",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--log_lvl",
        default=logging.INFO,
        help="Log level (default: %(default)s)",
        required=False,
        type=int,
    )

    # parse
    args = parser.parse_args()

    # set log level
    custom_log.init_logger(log_lvl=args.log_lvl)

    # run
    LOG.info("Start File Generation")
    create_files(b_path=args.directory, doe_name=args.doe_name)
    LOG.info("Files Generated")


if __name__ == "__main__":
    main()
