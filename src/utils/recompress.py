import argparse
import logging
import multiprocessing
import zipfile
import zlib
from pathlib import Path
from typing import Tuple

import custom_log

LOG: logging.Logger = logging.getLogger(__name__)


def eval_cmd_line() -> Tuple[Path, bool]:
    """Evaluate command line

    Returns:
        Tuple[Path, bool]: Path to work in, verbosity
    """
    # init
    parser = argparse.ArgumentParser(description="Load data for DOE2FE")

    # arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Directory to fetch data from",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        help="Number of cores to use",
        required=False,
        default=1,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    # parse
    args = parser.parse_args()
    custom_log.init_logger(log_lvl=logging.DEBUG if args.verbose else logging.INFO)

    return args.directory, args.cores


def get_dir_size_mb(path: Path, exp: int = 2) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**exp)


def run_parallel(sim_dir: Path):
    # log = custom_log.init_logger(__name__, log_lvl=custom_log.LEVELS.INFO)
    # log.info("Process %s, %s/%s", sim_dir)
    dir_size_before = get_dir_size_mb(sim_dir)

    # re-compress
    zip_file = sim_dir / "binouts.zip"
    with zipfile.ZipFile(file=zip_file) as archive:
        content_names = archive.namelist()
        # LOG.debug("Extract %s files from %s", len(content_names), zip_file.relative_to(sim_dir))
        archive.extractall(path=sim_dir)
    with zipfile.ZipFile(
        file=zip_file,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=zlib.Z_BEST_COMPRESSION,
    ) as archive:
        for binout in content_names:
            # log.debug("Compress %s to %s", binout, zip_file.relative_to(sim_dir))
            archive.write(sim_dir / binout, binout)

    # clean directory
    files = sim_dir.glob("*")
    for zip_file in files:
        if zip_file.suffix not in {".zip", ".key"}:
            if zip_file.is_file():
                # log.info("Remove %s", zip_file.relative_to(sim_dir))
                zip_file.unlink()
            else:
                for fi in zip_file.glob("*"):
                    # log.info("Remove %s", fi.relative_to(sim_dir))
                    fi.unlink()
                zip_file.rmdir()

    dir_size_after = get_dir_size_mb(sim_dir)
    fact = 100 * ((dir_size_before - dir_size_after) / dir_size_before)
    # log.info("Directory size before %.2fMB, after %.2fMB - %.1fPerc", dir_size_before, dir_size_after, fact)


def main(b_path: Path, cores: int):
    sim_dirs = sorted(b_path.glob("V*"))
    LOG.warning("Found %s simulation directories in %s", len(sim_dirs), b_path)
    total_size_before = get_dir_size_mb(b_path, exp=3)
    cores_availible = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cores if cores <= cores_availible else cores_availible) as pool:
        pool.map(run_parallel, sim_dirs)

    total_size_after = get_dir_size_mb(b_path, exp=3)
    fact = 100 * ((total_size_before - total_size_after) / total_size_before)
    LOG.warning("Directory size before %.2fGB, after %.2fGB - %.1fPerc", total_size_before, total_size_after, fact)


if __name__ == "__main__":
    main(*eval_cmd_line())
