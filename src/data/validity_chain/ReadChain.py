import argparse
import datetime
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src.data.validity_chain.ReadModelSims import ReadModelSims
from src.data.validity_chain.ReadReportData import ReadReportData
from src.utils.Csv import Csv
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


class ReadChain:
    def __init__(self, base_dir: Path, read_new: bool = False) -> None:
        # init
        checker = PathChecker()
        self.__read_new: bool = read_new

        # to read
        dummies = ["TH2.1", "TH2.7", "H3Rigid"]
        self.__cases: Dict[str, str] = {
            # "Honda_Accord_2014_Original": dummies[0],
            "Honda_Accord_2014_Original_THOR_2_7": dummies[1],
            # "Honda_Accord_2014_Original_THOR_2_7_Update_PAB": dummies[1],
            "Honda_Accord_2014_Original_with_HIII": dummies[2],
            # "Honda_Accord_2014_Sled_THOR_2_7": dummies[1],
            # "Honda_Accord_2014_Sled_with_HIII": dummies[2],
            # "Honda_Accord_2014_Sled_with_HIII_Rigid_Seat": dummies[2],
            "Honda_Accord_2014_Sled_with_HIII_RuntimeMin": dummies[2],
        }

        # env
        self.__base_dir: Path = checker.check_directory(path=base_dir, exit=True)
        self.__case_dirs: List[Path] = []
        for case in self.__cases.keys():
            case_dir = self.__base_dir / case
            if checker.check_directory(path=case_dir, exit=False):
                self.__case_dirs.append(case_dir)
        LOG.info("Found %s cases in %s", len(self.__case_dirs), self.__base_dir)

        # reports
        self.__report_dir = checker.check_directory(path=self.__base_dir / "From_Reports" / "csvs", exit=True)

    def run(self) -> pd.DataFrame:
        LOG.info("Start reading data")

        # read report
        LOG.info("Start parsing report CSVs in %s", self.__report_dir)
        reader = ReadReportData(in_dir=self.__report_dir, out_dir=self.__report_dir.parent)
        reader.run()
        reader.store(db_name="extracted", compress=True)
        data = [reader.data]
        LOG.info("Got data in shape %s from report CSVs in %s", data[-1].shape, self.__report_dir)

        # read simulations
        LOG.info("Start reading FE simulations")
        for case_dir in self.__case_dirs:
            LOG.info("Process %s", case_dir)
            dummy_version = self.__cases[case_dir.stem]
            reader = ReadModelSims(root_dir=case_dir, dummy_version=dummy_version, read_new=self.__read_new)
            data.append(reader.read_data())
            reader.store(sim_data=data[-1])
            LOG.info("Got data in shape %s from simulations in %s", data[-1].shape, case_dir)

        # combine data
        LOG.info("Combine %s dataframes", len(data))
        data = pd.concat(data, ignore_index=True)

        data["Case"].replace(
            to_replace={
                "full_frontal_56kmh": "Full Frontal",
                "oblique_left_90kmh": "Oblique Left",
                "oblique_right_90kmh": "Oblique Right",
                "01_Full_Frontal": "Full Frontal",
                "Oblique_Left": "Oblique Left",
                "Oblique_Right": "Oblique Right",
                "Full_Frontal": "Full Frontal",
                "Full_Frontal_DR": "Full Frontal",
                "Full_Frontal_PA": "Full Frontal",
                "Oblique_Left_DR": "Oblique Left",
                "Oblique_Left_PA": "Oblique Left",
                "Oblique_Right_DR": "Oblique Right",
                "Oblique_Right_PA": "Oblique Right",
            },
            inplace=True,
        )
        LOG.info("Got data in shape %s from %s", data.shape, self.__base_dir)

        return data

    def store(self, db: pd.DataFrame):
        # write csv
        LOG.info("Store Data of shape %s", db.shape)
        csv = Csv(csv_path=self.__base_dir / "extracted", compress=True)
        csv_path = csv.write(db)
        csv_hash = hash_file(fpath=csv_path)

        # get sub info
        sub_dirs = self.__case_dirs + [self.__report_dir.parent]
        data_infos = {str(fp.absolute()): json_util.load(f_path=fp / "data_info") for fp in sub_dirs}

        # store info
        LOG.info("Store data info")
        json_fpath = json_util.dump(
            obj={
                "Creation": str(datetime.datetime.now()),
                "Input": data_infos,
                "Output": {
                    "Directory": self.__base_dir.absolute(),
                    "Database": csv_path.name,
                    "Data Hash": csv_hash,
                },
            },
            f_path=self.__base_dir / "data_info",
        )

        LOG.info("Data stored")

        # copy into project directory
        project_dir = Path("data") / "validity_chain"
        project_dir.mkdir(parents=True, exist_ok=True)
        for fpath in [csv_path, json_fpath]:
            LOG.info("Copy '%s' to %s", fpath, project_dir)
            shutil.copy2(src=fpath, dst=project_dir)


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
        "--read_new",
        required=False,
        default=True,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Read data fresh from scratch if True",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=10,
        type=int,
        help="Log level (default is %(default)s)",
    )

    return parser.parse_args()


def run() -> None:
    # command line
    args = eval_cmd()

    # log
    custom_log.init_logger(log_lvl=args.log_lvl)
    LOG.debug("Command line input: %s", args)

    # run
    reader = ReadChain(base_dir=args.directory, read_new=args.read_new)
    data = reader.run()
    reader.store(db=data)
    LOG.info("DONE")


if __name__ == "__main__":
    run()
