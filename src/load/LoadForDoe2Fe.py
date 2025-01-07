import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import psutil
import logging

import numpy as np
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
import src.utils.hash_file as hash_file
import src.utils.json_util as json_util
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)


class LoadFromBigDoe:
    def __init__(
        self, in_dir: Path, out_dir: Path, side: Literal["01", "03"] = "03", cfc: Literal["A", "B", "C", "D", "X"] = "D"
    ):
        """Load data from big DOE

        Args:
            in_dir (Path): raw data directory
            out_dir (Path): processed data directory
            side (Literal[&quot;01&quot;, &quot;03&quot;], optional): driver/passenger. Defaults to "03".
            cfc (Literal[&quot;A&quot;, &quot;B&quot;, &quot;C&quot;, &quot;D&quot;, &quot;X&quot;], optional): cfc filter class. Defaults to "D".
        """
        self.str: StandardNames = StandardNames()
        f_in_end: str = ".parquet"
        f_out_end: str = ".npy"

        # directories
        self.raw_data_dir: Path = self.__check_path(in_dir)
        self.processed_data_dir: Path = self.__check_path(out_dir)

        # input files
        self.in_doe_path: Path = self.__check_path(self.raw_data_dir / f"doe{f_in_end}")
        self.in_channels_path: Path = self.__check_path(self.raw_data_dir / f"channels{f_in_end}")
        self.in_inj_val_path: Path = self.__check_path(self.raw_data_dir / f"injury_criteria{f_in_end}")

        # output files
        percentiles = (5, 50, 95)
        self.out_inj_vals_paths: Dict[int, Path] = {
            p: self.processed_data_dir / f"injury_values_tabular_{p:02}th{f_out_end}" for p in percentiles
        }
        self.out_feature_3d_paths: Dict[int, Path] = {
            p: self.processed_data_dir / f"channels_3D_{p:02}th{f_out_end}" for p in percentiles
        }
        self.out_does: Dict[int, Path] = {
            p: self.processed_data_dir / f"{self.in_doe_path.stem}{p:02}th{f_out_end}" for p in percentiles
        }

        # misc
        self.side: Literal["01", "03"] = side
        self.cfc: Literal["A", "B", "C", "D", "X"] = cfc
        self.ids: Dict[int, List[int]] = {}
        self.time_stamps: List[float] = []

        # relevant channels
        self.channel_feature = [
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03HEAD0000OCCUACXD",
            "03HEAD0000OCCUACYD",
            "03HEAD0000OCCUACZD",
            "03CHSTLOC0OCCUDSXD",
            "03CHSTLOC0OCCUDSYD",
            "03CHSTLOC0OCCUDSZD",
            "03CHST0000OCCUDSXD",
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACYD",
            "03CHST0000OCCUACZD",
            "03PELVLOC0OCCUDSXD",
            "03PELVLOC0OCCUDSYD",
            "03PELVLOC0OCCUDSZD",
            "03PELV0000OCCUACXD",
            "03PELV0000OCCUACYD",
            "03PELV0000OCCUACZD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03FEMRRI00OCCUFOZD",
            "03FEMRLE00OCCUFOZD",
            "03FACE0000OCCUFORD",
            "03SHLDRIUPOCCUFORD",
            "03SHLDLEUPOCCUFORD",
            "03SHLDRILOOCCUFORD",
            "03SHLDLELOOCCUFORD",
            "03CHSTRIUPOCCUFORD",
            "03CHSTLEUPOCCUFORD",
            "03CHSTRILOOCCUFORD",
            "03CHSTLELOOCCUFORD",
            "03KNEERI00OCCUFORD",
            "03KNEELE00OCCUFORD",
        ]
        self.channel_target = [
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03HEAD0000OCCUACXD",
            "03HEAD0000OCCUACYD",
            "03HEAD0000OCCUACZD",
            "03CHSTLOC0OCCUDSXD",
            "03CHSTLOC0OCCUDSYD",
            "03CHSTLOC0OCCUDSZD",
            "03CHST0000OCCUDSXD",
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACYD",
            "03CHST0000OCCUACZD",
            "03PELVLOC0OCCUDSXD",
            "03PELVLOC0OCCUDSYD",
            "03PELVLOC0OCCUDSZD",
            "03PELV0000OCCUACXD",
            "03PELV0000OCCUACYD",
            "03PELV0000OCCUACZD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03FEMRRI00OCCUFOZD",
            "03FEMRLE00OCCUFOZD",
        ]

    def process_doe(self):
        """process DOE data"""
        # read
        id_str = "SIM_ID"
        LOG.info("Read DOE from %s", self.in_doe_path)
        db = pd.read_parquet(self.in_doe_path)
        db.set_index("PERC", inplace=True, append=True)
        LOG.debug("DOE shape: %s\n%s", db.shape, db)
        cols = sorted(db.columns)
        db_idx = db.reset_index()
        db_idx = db_idx.set_index(sorted(set(db_idx.columns) - {id_str}))

        # process
        for perc in self.out_does.keys():
            # split
            LOG.info("Process %s th Percentile", perc)
            db_perc = db.loc[(slice(None), perc), cols]
            LOG.debug("DOE shape: %s", db_perc.shape)
            idxs = db_idx.loc[(slice(None), slice(None), perc, slice(None), slice(None)), id_str]
            idxs = idxs.droplevel("PERC").sort_index()
            self.ids[perc] = idxs.to_list()
            db_perc = db.loc[(self.ids[perc], perc), cols]
            LOG.debug("IDs: %s", len(self.ids[perc]))

            # store
            as_np = db_perc.to_numpy()
            LOG.info("Store DOE %s of shape %s", self.out_does[perc], as_np.shape)
            np.save(self.out_does[perc], as_np)
            self.write_data_info(in_file=self.in_doe_path, out_file=self.out_does[perc], axes_labels=[self.ids[perc], cols])

        # check
        does = None
        same = True
        for perc in self.out_does.keys():
            new = np.load(self.out_does[perc])
            if does is None:
                does = new
            else:
                same = same and np.allclose(does, new)
        if same:
            LOG.info("DOE data identical")
        else:
            LOG.warning("All DOE data not identical - OK")

        LOG.info("DOE data processed")

    def process_injury_values(self):
        """process injury values"""
        LOG.info("Read injury values from %s", self.in_inj_val_path)
        db = pd.read_parquet(self.in_inj_val_path)
        LOG.debug("Injury values shape: %s\n%s", db.shape, db)

        columns = sorted(db.columns)

        for perc in self.ids.keys():
            # split
            LOG.info("Process %s th Percentile", perc)
            inj_vals = db.loc[self.ids[perc], columns].to_numpy()

            # store
            LOG.info("Store injury values %s of shape %s", self.out_inj_vals_paths[perc], inj_vals.shape)
            np.save(self.out_inj_vals_paths[perc], inj_vals)
            LOG.info("Injury values stored: %sMB", os.stat(self.out_inj_vals_paths[perc]).st_size / (1024 * 1024))
            self.write_data_info(
                in_file=self.in_inj_val_path, out_file=self.out_inj_vals_paths[perc], axes_labels=[self.ids[perc], columns]
            )

        LOG.info("Injury values processed")

    def process_channels(self):
        """process channels data"""
        LOG.info("Read channels from %s", self.in_channels_path)
        db = pd.read_parquet(self.in_channels_path)
        LOG.debug("Channels shape: %s\n%s", db.shape, db)

        self.time_stamps = sorted(db.index.get_level_values("TIME").unique())

        for perc in self.ids.keys():
            if perc == 50:
                channels = self.channel_feature
            else:
                channels = self.channel_target
            channels = sorted([c for c in channels if c in set(db.columns)])
            LOG.info("Selected number of channels: %s", len(channels))
            # split
            LOG.info("Process %s th Percentile", perc)
            as_np = db.loc[(self.ids[perc], self.time_stamps), channels].to_xarray().to_array().values
            as_np = as_np.transpose(1, 0, 2)
            LOG.debug("Channels shape: %s", as_np.shape)

            # store
            LOG.info("Store channels %s of shape %s", self.out_feature_3d_paths[perc], as_np.shape)
            np.save(self.out_feature_3d_paths[perc], as_np)
            LOG.info("Channels stored: %sMB", os.stat(self.out_feature_3d_paths[perc]).st_size / (1024 * 1024))

            # reduce storage usage
            LOG.debug("Reduce storage usage from %sGB", psutil.Process(os.getpid()).memory_info()[0] / 2.0**30)
            del as_np
            db = db.drop(self.ids[perc], level=0, axis=0)
            LOG.debug("Reduce storage usage to %sGB", psutil.Process(os.getpid()).memory_info()[0] / 2.0**30)

            # bookkeeping
            self.write_data_info(
                in_file=self.in_channels_path,
                out_file=self.out_feature_3d_paths[perc],
                axes_labels=[self.ids[perc], channels, self.time_stamps],
            )

        LOG.info("Channels processed")

    def __check_path(self, path: Path) -> Path:
        """Check if path exists

        Args:
            path (Path): path to check

        Raises:
            FileNotFoundError: if path does not exist

        Returns:
            Path: existing path
        """
        if path.exists():
            LOG.debug("Path exists: %s OK", path)
            return path
        else:
            raise FileNotFoundError(f"Not found: {path}")

    def write_data_info(self, in_file: Path, out_file: Path, axes_labels: List[list]):
        """Write information about data to disk

        Args:
            in_file (Path): input file
            out_file (Path): stored file
            axes_labels (List[list]): array axis labels
        """
        # content
        book = {
            self.str.creation: str(datetime.datetime.now()),
            self.str.input: {
                self.str.path: in_file,
                self.str.hash: hash_file.hash_file(in_file),
            },
            self.str.output: {
                self.str.path: out_file,
                self.str.hash: hash_file.hash_file(out_file),
            },
            self.str.axis: axes_labels,
        }

        # store
        info_path = out_file.parent / f"{out_file.stem}_info"
        LOG.info("Store info to %s", info_path)
        json_util.dump(obj=book, f_path=info_path)


def main():
    """run main function"""
    in_directory, out_directory = eval_cmd_line()
    LOG.info("Load data")

    loader = LoadFromBigDoe(in_dir=in_directory, out_dir=out_directory)
    loader.process_doe()
    loader.process_channels()
    loader.process_injury_values()


def eval_cmd_line() -> Tuple[Path, Path]:
    """Evaluate and check command line arguments

    Raises:
        FileNotFoundError: one of the directories does not exist

    Returns:
        Tuple[Path, Path]: raw data directory, processed data directory
    """
    # init
    parser = argparse.ArgumentParser(description="Load data for DOE2FE")

    # arguments
    parser.add_argument(
        "-i",
        "--in_directory",
        type=Path,
        help="Directory to fetch data from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_directory",
        type=Path,
        help="Directory to store data to",
        required=False,
        default=None,
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

    # check input directory
    if args.in_directory.is_dir():
        LOG.info("Input directory: %s", args.in_directory)
    else:
        LOG.critical("Input directory not found: %s", args.in_directory)
        raise FileNotFoundError(f"Input directory not found: {args.in_directory}")

    # check output directory
    if args.out_directory.is_dir():
        LOG.warning("Output directory exists and will be overwritten: %s", args.out_directory)
        shutil.rmtree(args.out_directory)
    else:
        LOG.info("Output directory: %s", args.out_directory)
    args.out_directory.mkdir(parents=True, exist_ok=True)

    return args.in_directory, args.out_directory


if __name__ == "__main__":
    main()
