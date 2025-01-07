import datetime
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

src_dir = str(Path(__file__).absolute().parents[3])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.json_util as json_util
from src.utils.Csv import Csv
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)


class ReadGeneral:
    def __init__(self, in_dir: Path, out_dir: Union[Path, None] = None) -> None:
        self.checker = PathChecker()

        # set input directory
        self.dir = self.checker.check_directory(path=in_dir, exit=True)

        # set output directory
        if out_dir is None:
            self.out_fpath = in_dir
            LOG.info("Store data in %s", self.out_fpath)
        else:
            self.out_fpath = self.checker.check_directory(path=out_dir, exit=True)

        # init
        self.data = pd.DataFrame()
        self.in_files: List[Path] = []
        self.in_hash: str = ""
        self.cfcs = [None, 60, 180, 600, 1000]
        self.cfc_names = {
            1000: "A",
            600: "B",
            180: "C",
            60: "D",
            None: "X",
        }  # ISO 6487* / SAE J211:MAR95 1.0

        # standard names
        ff, obl, obr = "Full Frontal", "Oblique Left", "Oblique Right"
        self.standard_cases = {
            "full_frontal_56kmh": ff,
            "oblique_left_90kmh": obl,
            "oblique_right_90kmh": obr,
            "01_Full_Frontal": ff,
            "Oblique_Left": obl,
            "Oblique_Right": obr,
            "Full_Frontal": ff,
            "Full_Frontal_DR": ff,
            "Full_Frontal_PA": ff,
            "Oblique_Left_DR": obl,
            "Oblique_Left_PA": obl,
            "Oblique_Right_DR": obr,
            "Oblique_Right_PA": obr,
        }

        self.channels: List[str] = []

    def store(self, db_name: str, compress: bool = True):
        """Store data and properties

        Args:
            db_name (str): stem of name of database
        """

        # store database
        csv = Csv(csv_path=self.out_fpath / db_name, compress=compress)
        csv_path: Path = csv.write(self.data)
        csv_hash = hash_file(fpath=csv_path)

        # in info
        in_hash = self.in_hash if self.in_hash else hash_file(fpath=self.in_files)

        # store info
        json_util.dump(
            obj={
                "Creation": f"{datetime.datetime.now()} {datetime.datetime.now().astimezone().tzname()}",
                "Input": {
                    "Directory": self.dir.absolute(),
                    "Files": [x.name for x in self.in_files],
                    "Data Hash": in_hash,
                },
                "Output": {
                    "Directory": self.out_fpath.absolute(),
                    "Database": csv_path.name,
                    "Data Hash": csv_hash,
                },
            },
            f_path=self.out_fpath / "data_info",
        )

    def __get_couples(self) -> List[List[str]]:
        """Get channels only differing in their direction

        Returns:
            List[List[str]]: channels combinations
        """
        couples = defaultdict(set)
        channels = set(self.channels)
        for string_a in channels:
            # exclude resultants
            res_name = string_a[:-2] + "R" + string_a[-1]
            if res_name not in channels:
                for string_b in channels:
                    if string_a[:-2] == string_b[:-2] and string_a[-1] == string_b[-1]:
                        # allow difference at string[-2] (MME direction)
                        couples[string_a].add(string_b)

        # return sorted combinations
        return [sorted(x) for x in couples.values()]

    def _add_resultants(self, db: pd.DataFrame) -> pd.DataFrame:
        combinations = self.__get_couples()

        db_new = db.copy()
        for comb in combinations:
            res_name = comb[0][:-2] + "R" + comb[0][-1]
            db_new[res_name] = np.linalg.norm(db[comb], axis=1)

        return db_new
