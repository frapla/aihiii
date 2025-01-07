import argparse
import datetime
import shutil
import sys
from pathlib import Path
from typing import Dict, List
import logging

import pandas as pd

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src.utils.CfCFilter import CfCFilter
from src.utils.hash_file import hash_file
from src.utils.PathChecker import PathChecker
from src.utils.ReadBinout import ReadBinout
from src.utils.UnifySignal import UnifySignal

LOG: logging.Logger = logging.getLogger(__name__)


class GetPulse:
    def __init__(
        self, in_dir: Path, target_tend_ms: float = 160, target_sampling_rate_ms: float = 0.1, out_dir_name: str = "PULSE"
    ) -> None:
        # parameter
        self.cfc: int = 60
        self.locs_nids: Dict[str, str] = {
            "B_PillarSillRight": "82000020",
            "RearSillPassenger": "82000004",
            "B_PillarSillLeft": "82000028",
            "RearSillDriver": "82000012",
        }

        # in directory
        self.in_dir: Path = PathChecker().check_directory(in_dir)
        self.out_dir: Path = self.in_dir / out_dir_name
        self.out_dir.mkdir(exist_ok=True)
        self.info_file: Path = self.out_dir / "info.json"
        self.pulse_file: Path = self.out_dir / "pulse.k"
        self.final_dir: Path = (self.in_dir.parent.parent.parent / "000_PULSE_INTRUSION") / self.in_dir.relative_to(self.in_dir.parent.parent)
        self.ref_path: Path = self.in_dir.relative_to(self.in_dir.parent.parent.parent)

        # init
        self.cfc_filter: CfCFilter = CfCFilter()
        self.unifier: UnifySignal = UnifySignal(
            target_tend_ms=target_tend_ms,
            target_sampling_rate_ms=target_sampling_rate_ms,
            padwith=0,
        )
        self.data: Dict[str, pd.DataFrame] = {}
        self.tsp = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")

    def process(self) -> None:
        LOG.info("Read binout files from %s", self.in_dir)
        self.data = self.__read_binout()

        LOG.info("Unify and filter data")
        self.data = self.__unify_filter()

        LOG.info("Convert to pulse file")
        body = self.__data_2_pulse_file()

        LOG.info("Write pulse file")
        self.__write_pulse_file(body)

        LOG.info("Bookeeping")
        self.__bookeeping()

        LOG.info("Copy to final directory")
        self.__copy_2_end()

        LOG.info("Done")

    def __copy_2_end(self):
        self.final_dir.mkdir(parents=True, exist_ok=True)
        for file in self.out_dir.glob("*"):
            LOG.debug("Copy %s to %s", file, self.final_dir / file.name)
            shutil.copy(src=file, dst=self.final_dir / file.name)

    def __read_binout(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        with ReadBinout(sim_dir=self.in_dir) as binout:
            for direction in ("x", "y", "z"):
                # get data
                data[direction] = binout.binout.as_df("nodout", f"{direction}_acceleration")[sorted(self.locs_nids.values())]
                LOG.debug("Got %s data for direction %s", data[direction].shape, direction)

        return data

    def __unify_filter(self) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for direction in self.data.keys():
            # convert time
            db = self.data[direction].copy()
            db.index *= 1000  # s to ms

            # unify and zero pad
            db = self.unifier.unify(db=db)
            LOG.debug("Unified data to %s", db.shape)

            # filter CFC
            db = pd.DataFrame(
                self.cfc_filter.filter(
                    tsp=0.001 * self.unifier.target_sampling_rate_ms, signal=db.values, cfc=self.cfc, force_zero=True
                ),
                index=db.index,
                columns=db.columns,
            )
            LOG.debug("Filtered data shape %s with CFC%s", db.shape, self.cfc)

            # store
            db.index *= 0.001  # ms to s
            data[direction] = db

        return data

    def __data_2_pulse_file(self) -> List[str]:
        body = [
            f"$# Pulse file created {self.tsp}",
            f"$# from {self.ref_path / 'binout*'}",
            "*KEYWORD",
            "*TITLE",
            "$#",
            f"Pulse {self.in_dir.stem}",
        ]
        define_start = "*DEFINE_CURVE_TITLE"
        comment_1 = "$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint"
        comment_2 = "$#                a1                  o1"
        sep = "$" + "-" * 80
        inv_locs_nids = {v: k for k, v in self.locs_nids.items()}

        for nid in sorted(self.locs_nids.values()):
            body.append(sep)
            for i, direction in enumerate(sorted(self.data.keys())):
                body.extend(
                    [
                        define_start,
                        f"{direction}_acceleration@{nid} {inv_locs_nids[nid]}",
                        comment_1,
                        f"{int(nid)+i:>10}         0       1.0&vehxscal        0.0       0.0         0{self.data[direction].shape[0]:>10}",
                        comment_2,
                    ]
                )
                for tsp in self.data[direction].index:
                    body.append(f"{tsp:>20.10e}{self.data[direction].loc[tsp, nid]:>20.10e}")
        body.append("*END")

        return body

    def __write_pulse_file(self, body: List[str]) -> None:
        LOG.debug("Write pulse file to %s", self.pulse_file)
        with open(self.pulse_file, "w") as f:
            [f.write(f"{x}\n") for x in body]
        LOG.debug("Wrote pulse filewith %s lines to %s", len(body), self.pulse_file)

    def __bookeeping(self) -> None:
        info = {
            "tsp": self.tsp,
            "in_dir": self.ref_path,
            "cfc": self.cfc,
            "locs_nids": self.locs_nids,
        }

        # hash pulse file
        info["Pulse"] = {"File": self.pulse_file.relative_to(self.in_dir), "Hash": hash_file(self.pulse_file)}

        # hash more files
        for f_type in ["binout", "d3plot", "mes"]:
            files = sorted(self.in_dir.glob(f"{f_type}*"))
            info[f_type] = {"Files": [x.relative_to(self.in_dir) for x in files], "Hash": hash_file(files)}

        # hash interface
        f_type = "interface"
        files = sorted(self.out_dir.glob(f"{f_type}*"))
        info[f_type] = {"Files": [x.relative_to(self.in_dir) for x in files], "Hash": hash_file(files)}

        # store
        json_util.dump(obj=info, f_path=self.info_file)


def eval_cmd_line() -> Path:
    # cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        required=True,
        type=Path,
        help="Path to directory with simulation directories",
    )
    parser.add_argument(
        "--log_lvl",
        required=False,
        default=20,
        type=int,
        help="Log level (default is %(default)s)",
    )
    args = parser.parse_args()

    # init logger
    custom_log.init_logger(log_lvl=logging.DEBUG)

    # set path
    in_dir = PathChecker().check_directory(args.in_dir)

    return in_dir


if __name__ == "__main__":
    reader = GetPulse(in_dir=eval_cmd_line())
    reader.process()
    LOG.info("%s Done", __name__)
