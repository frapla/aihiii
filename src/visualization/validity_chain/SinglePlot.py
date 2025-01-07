import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).absolute().parents[3]
print(PROJECT_DIR)
if str(PROJECT_DIR) not in set(sys.path):
    sys.path.append(str(PROJECT_DIR))
import src.utils.custom_log as custom_log
from src.utils.Csv import Csv
from src.visualization.validity_chain.StandardStr import StandardStr as Str

LOG: logging.Logger = logging.getLogger(__name__)


class SinglePlot:
    def __init__(
        self,
        data: pd.DataFrame,
        confs: List[str],
        ch: str,
        case: str,
        side: str,
        ax: plt.Axes,
    ) -> None:
        # data
        self.__db: pd.DataFrame = data
        LOG.info(f"Dataframe shape: {self.__db.shape}")
        self.__db_rep: Optional[pd.DataFrame] = None
        self.__db_sim: Dict[str, pd.DataFrame] = {}

        # filter settings
        if isinstance(confs, list):
            self.__confs: List[str] = [self.__check_filter_str(f_str=conf, col=Str.configuration) for conf in confs]
        else:
            LOG.critical("confs must be a list of strings - got %s - EXIT", type(confs))
            sys.exit(1)
        self.__ch: str = self.__check_filter_str(f_str=ch, col=Str.channel)
        self.__case: str = self.__check_filter_str(f_str=case, col=Str.case)
        self.__side: str = self.__check_filter_str(f_str=side, col=Str.side)

        # helpers
        self.__iso_2_unit_str: Dict[str, str] = {"FO": "Force [kN]", "AC": "Acceleration [g]", "VE": "Velocity [m/s]"}
        self.__assemblies: List[str] = sorted(self.__db[Str.assembly].dropna().unique())
        self.__assemblies_48cpu: List[str] = ["Assemblies_1", "Assemblies_3"]
        self.__assemblies_96cpu: List[str] = ["Assemblies_2"]
        self.__ass_dict = {"Assemblies_1": 48, "Assemblies_2": 96, "Assemblies_3": 48}
        self.__cases: List[str] = sorted(self.__db[Str.case].dropna().unique())
        self.__channels: List[str] = sorted(self.__db[Str.channel].dropna().unique())
        self.__sides: List[str] = sorted(self.__db[Str.side].dropna().unique())
        self.__configs: List[str] = sorted(self.__db[Str.configuration].dropna().unique())
        self.__sources: List[str] = sorted(self.__db[Str.source].dropna().unique())

        # plot
        self.__ax: plt.Axes = ax
        self._fill_color: str = "gray"
        self._assembly_colors: List[str] = ["gray", "gray", "gray"]
        self._median_color: str = "black"
        self._test_color: str = "black"
        self._cae_color: str = "red"
        self.__colors = list(mcolors.TABLEAU_COLORS.values())

    def create(self):
        """Create plot"""
        self.__filter_db()
        self.__plot()

    def __check_filter_str(self, f_str: str, col: str) -> Optional[str]:
        """Check if filter string is in column.

        Args:
            f_str (str): element to filter with
            col (str): column to filter

        Returns:
            str:  element to filter with
        """
        if f_str in self.__db[col].unique():
            return f_str
        else:
            LOG.warning(f"Filter string {f_str} not found in {col}.")
            return None

    def __filter_db(self) -> None:
        """Filter database"""
        # fitler simulation data
        for conf in self.__confs:
            # filter configuration
            db_config = self.__db[self.__db[Str.configuration].eq(conf)].copy()
            LOG.debug("Filter config %s to shape %s", conf, db_config.shape)

            # filter side
            db_side = db_config[db_config[Str.side].isna() | db_config[Str.side].eq(self.__side)]
            LOG.debug("Filter side %s to shape %s", self.__side, db_side.shape)

            # filter case
            db_case = db_side[db_side[Str.case].eq(self.__case)]
            LOG.debug("Filter case %s to shape %s", self.__case, db_case.shape)

            # filter channel
            if self.__ch in set(db_case[Str.channel].to_list()):
                ch = self.__ch
                LOG.debug("Channel %s found in db.", ch)
            elif "H3" in self.__ch:
                ch = self.__ch.replace("H3", "TH")
                LOG.debug("Channel %s not found in db. Try %s", self.__ch, ch)
            else:
                ch = self.__ch.replace("TH", "H3")
                LOG.debug("Channel %s not found in db. Try %s", self.__ch, ch)
            db_ch = db_case[db_case[Str.channel].eq(ch)]
            LOG.debug("Filter channel %s to shape %s", ch, db_ch.shape)

            # store
            self.__db_sim[conf] = db_ch

        # filter report data
        db_rep = self.__db[self.__db[Str.configuration].isna()].copy()
        self.__db_rep = db_rep[db_rep[Str.channel].eq(self.__ch.replace("H3", "TH")) & db_rep[Str.case].eq(self.__case)]
        LOG.debug("Filter report data to shape %s", self.__db_rep.shape)

    def __plot(self) -> None:
        """Create plot"""
        LOG.info("Create plot.")
        for src in self.__sources:
            LOG.info("Add %s data.", src)
            if src != "CAE THI":
                self.__plot_report_data(src=src)
        for src in self.__sources:
            LOG.info("Add %s data.", src)
            if src == "CAE THI":
                self.__plot_simulation_data(src=src)

        self.__format_plot()

    def __plot_simulation_data(self, src: str) -> None:
        """Plot data from own simulations

        Args:
            src (str): should be invariant
        """
        LOG.info("Add simulation data.")
        for k, conf in enumerate(self.__confs):
            # TODO name nicely
            conf_short: str = conf

            # flip data y
            flip_sim = (
                -1
                if ("FEMR" in self.__ch and "TH50" in self.__ch and "2_7" not in conf)
                or ("FEMR" in self.__ch and "H350" in self.__ch)
                or (self.__ch == "00COG00000VH00ACXC")
                else 1
            )
            LOG.debug("Multiply simulation data y of %s, %s by %s", conf, self.__ch, flip_sim)

            # get plot data
            # for o, assemblies in enumerate([self.__assemblies_48cpu, self.__assemblies_96cpu]):
            db_source = self.__db_sim[conf][self.__db_sim[conf][Str.source].eq(src)]
            new_db = []
            for ass in self.__assemblies:
                LOG.debug("Add %s", ass)
                db_ass = db_source[db_source[Str.assembly].eq(ass)]
                new_db.append(
                    pd.Series(
                        data=db_ass[Str.value].replace({db_ass["Value"].to_list()[-1]: np.nan}).to_numpy() * flip_sim,
                        index=db_ass[Str.time].to_numpy(),
                        name=ass,
                    )
                )
            new_db = pd.concat(new_db, axis=1)
            """
                # get corridor data
                new_db[Str.median] = new_db[assemblies].median(axis=1)
                new_db[Str.min] = new_db[assemblies].min(axis=1)
                new_db[Str.max] = new_db[assemblies].max(axis=1)

                # plot
                self.__ax.fill_between(
                    x=new_db.index,
                    y1=new_db[Str.min],
                    y2=new_db[Str.max],
                    label=f"{conf_short}",
                    alpha=0.5,
                    color=self.__colors[k + o],
                    edgecolor=self.__colors[k + o],
                )"""
            lss = ["--", "-.", ":"]
            for z, ass in enumerate(self.__assemblies):
                if "_3" not in ass:
                    self.__ax.plot(
                        new_db.index, new_db[ass], c=self.__colors[k], label=f"{conf_short} - {self.__ass_dict[ass]}", ls=lss[z]
                    )

    def __plot_report_data(self, src: str) -> None:
        """add data from validation report

        Args:
            src (str): description of type in validation report
        """
        LOG.info("Add report %s data.", src)
        db_source = self.__db_rep[self.__db_rep["Source"].eq(src)]
        flip = (
            -1
            if (self.__case == "Oblique Right" and self.__ch == "00COG00000VH00VEYC")
            or (self.__ch == "00COG00000VH00ACXC")
            or (self.__ch == "00COG00000VH00VEXC" and self.__case == "Full Frontal")
            else 1
        )
        self.__ax.plot(
            db_source["Time"],
            db_source["Value"] * flip,
            label=src,
            c=self._test_color if src.startswith("Test") else self._cae_color,
            alpha=0.7,
        )

    def __format_plot(self) -> None:
        """Format plot"""
        self.__ch = self.__ch.replace("FORC", "FOZC") if "FEMR" in self.__ch else self.__ch
        # 0 line
        self.__ax.axhline(0, c="black", alpha=0.5)

        # x axis
        self.__ax.set_xlabel("Time [ms]")
        self.__ax.set_xlim([0, 120])
        self.__ax.set_xticks(np.arange(0, 121, 10))

        # y axis
        unit = self.__iso_2_unit_str[self.__ch[-4:-2]]
        self.__ax.set_ylabel(unit)
        if "AC" in self.__ch:
            self.__ax.set_ylim([-20, 100])
            self.__ax.set_yticks(np.arange(-20, 101, 10))
        elif "FORC" in self.__ch:
            self.__ax.set_ylim([-1, 7])
            self.__ax.set_yticks(np.arange(-1, 8, 1))
        elif "FOZC" in self.__ch:
            self.__ax.set_ylim([-2, 6])
            self.__ax.set_yticks(np.arange(-2, 7, 1))
        elif "VE" in self.__ch:
            rng = [-6, 16] if self.__case == "Full Frontal" else [-16, 6]
            self.__ax.set_ylim(rng)
            self.__ax.set_yticks(np.linspace(*rng, 12))

        # general
        self.__ax.set_title(self.__ch)
        self.__ax.grid()
        self.__ax.legend()


def test():
    global PROJECT_DIR
    custom_log.init_logger(log_lvl=logging.DEBUG)
    _, ax = plt.subplots()
    csv = Csv(csv_path=PROJECT_DIR / "data" / "validity_chain" / "extracted", compress=True)
    db: pd.DataFrame = csv.read()

    plotter = SinglePlot(
        data=db,
        confs="Honda_Accord_2014_Original_with_HIII",
        ch="03HEAD0000H350ACRC",
        case="Oblique Left",
        side="DR",
        ax=ax,
    )
    plotter.create()
    plt.show(block=True)


if __name__ == "__main__":
    test()
