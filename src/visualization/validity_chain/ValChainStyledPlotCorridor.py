import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes as Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

sys.path.append(str(Path(__file__).absolute().parents[3]))
from src.utils.set_rcparams import set_rcparams

LOG: logging.Logger = logging.getLogger(__name__)


class ValChainStyledPlotCorridor:
    def __init__(
        self,
        channels: List[str],
        nb_path: Path,
        hw_ration: float,
        width_frac: Optional[float] = 1,
        fst_ratio: float = 0.02,
        override_format: bool = True,
        t_max: float = 130,
        heading: bool = False,
    ) -> None:
        """Creates plots with purpose of usage in dissertation

        Args:
            channels (List[str]): iso styled channel names
            nb_path (Path): directory to store data
            hw_ration (float): ratio of height to width
            width_frac (Optional[float], optional): fraction of page width. Defaults to 1.
            log_lvl (Optional[int], optional): logging level. Defaults to None.

        Raises:
            FileNotFoundError: mpl file not found
            FileNotFoundError: font directory not found
        """
        # plot formats
        set_rcparams()

        # storing format
        self.f_types: List[Literal["png", "pdf", "svg"]] = ["png", "pdf"]  # , "pdf", "svg"]

        # storing directory
        self.store_dir: Path = nb_path
        LOG.info("Store plots in %s with file extention(s) %s", self.store_dir, self.f_types)
        for ext in self.f_types:
            (self.store_dir / ext.upper()).mkdir(exist_ok=True)

        # sizing
        self.fig_width: float = width_frac * (448.13095 / 72)
        self.fig_height: float = hw_ration * self.fig_width
        self.fst_ratio: float = fst_ratio
        self.heading: bool = heading

        # stylers
        self.units: Dict[str, str] = {
            "FO": "Force [kN]",
            "AC": "Acceleration [g]",
            "VE": "Velocity [m/s]",
            "DS": "Length [mm]",
            "PR": "Pressure [bar]",
        }
        self.lss: List[str] = ["-", ":", "-.", "--"]
        self.cs_report: List[str] = ["black", "blue"]
        cmap = sns.color_palette("colorblind", as_cmap=True)
        self.fill_cs: List[str] = [cmap[1], cmap[2], cmap[4]]
        self.dummies: List[str] = ["TH", "H3"]
        self.override_format = override_format

        # ranges
        self.minis: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.maxis: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.spans: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.t_max: float = t_max

        # init
        self.fig: Figure
        self.axs: dict[Text, Axes]
        self.channels: List[str] = sum(channels, [])
        self.__init_plot(channel_mosaic=channels)

    def __init_plot(self, channel_mosaic: List[List[str]]) -> None:
        """Initializes plot with subplots and legend"""
        chs = [["none"] * len(channel_mosaic[0]), *channel_mosaic]
        self.fig, self.axs = plt.subplot_mosaic(
            chs,
            figsize=(self.fig_width, self.fig_height),
            gridspec_kw={"height_ratios": (self.fst_ratio, *([1] * len(channel_mosaic)))},
            layout="constrained",
        )
        self.axs["none"].axis("off")

    def format_store(
        self, region: str, case: Literal["Oblique Left", "Oblique Right", "Full Frontal"], side: Optional[int] = None
    ) -> None:
        """Stores plot

        Args:
            region (str): body part or vehicle
            case (Literal[&quot;Oblique Left&quot;, &quot;Oblique Right&quot;, &quot;Full Frontal&quot;]): load case
            side (Optional[int], optional): driver / passenger. Defaults to None.
        """
        # format
        if self.override_format:
            self.__format(suptitle=case)

        # store
        cs = "".join([c[0].upper() for c in case.split()])
        reg = region.upper()
        sd = {1: "DR", 3: "PA", None: ""}[side]
        for f_type in self.f_types:
            f_path = self.store_dir / f_type.upper() / f"{cs}_{sd}_{reg}.{f_type}"
            LOG.info(f"Store {f_path}")
            self.fig.savefig(f_path)

    def __format(self, suptitle: str) -> None:
        # add legend
        n_leg = np.argmax([len(self.axs[ch].get_legend_handles_labels()[1]) for ch in self.channels])
        self.axs["none"].legend(*self.axs[self.channels[n_leg]].get_legend_handles_labels(), loc="upper center", ncol=4)

        # add title
        if self.heading:
            self.fig.suptitle(suptitle)

        # each subplot
        for channel in self.channels:
            self.axs[channel].set_xlim([0, self.t_max])
            self.axs[channel].set_xticks(np.arange(0, self.t_max + 1, 10))
            self.axs[channel].set_title(channel[:-1] + "D")
            self.axs[channel].set_xlabel("Time [ms]")
            self.axs[channel].set_ylabel(self.units[self.get_dim(channel)])
            self.axs[channel].grid()

        # range y axis
        self.__auto_range()

    def __auto_range(self) -> None:
        """Set centered y ranges for all subplots"""
        ranges = self.__get_min_max_ranges()
        for dim in ranges.keys():
            for ch, span in ranges[dim].items():
                self.axs[ch].set_ylim(span)

    def add_line(self, channel: str, x: pd.Series, y: pd.Series, label: str, c_id: int, ls_id: Optional[int] = None) -> None:
        """Add line to plot

        Args:
            channel (str): channel name in iso style
            x (pd.Series): x axis values
            y (pd.Series): y axis values
            label (str): line label
            c_id (int): id in color wheel (>0 CAE wheel else report wheel)
            ls_id (Optional[int], optional): id in line style wheel. Defaults to None.
        """

        LOG.debug("Add %s", channel)
        # plot
        self.axs[channel].plot(
            x,
            y,
            label=label,
            c=self.fill_cs[c_id] if c_id >= 0 else self.cs_report[-c_id - 1],
            ls="-" if ls_id is None else self.lss[ls_id],
            lw=1 if label else 0.5,
        )

        # update ranges
        self.__update_min_max(channel=channel, vals=y)

    def add_range(self, cae: pd.DataFrame, channel: str, fill_c_id: int, label: str) -> None:
        LOG.debug("Add %s CAE corridor", channel)

        time = sorted(cae["Time"].unique())
        mini = [cae["Value"][cae["Time"].eq(t)].min() for t in time]
        maxi = [cae["Value"][cae["Time"].eq(t)].max() for t in time]
        self.axs[channel].fill_between(time, mini, maxi, color=self.fill_cs[fill_c_id], alpha=0.3, label=label, lw=0.5)

    def __update_min_max(self, channel: str, vals: pd.Series) -> None:
        """Store min and max values for each channel

        Args:
            channel (str): channel name in iso style
            vals (pd.Series): y axis values
        """
        dim = self.get_dim(channel)
        if len(vals) > 0:
            # replace nan with 0, else min/max would be messed up
            vals = vals.fillna(0).copy()

            # update min/max
            if channel not in self.minis[dim] or (channel in self.minis[dim] and min(vals) < self.minis[dim][channel]):
                self.minis[dim][channel] = min(vals)
            if channel not in self.maxis[dim] or (channel in self.maxis[dim] and max(vals) > self.maxis[dim][channel]):
                self.maxis[dim][channel] = max(vals)

            # update spans
            self.spans[dim][channel] = self.maxis[dim][channel] - self.minis[dim][channel]

    def __get_min_max_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get min and max values for total y axis

        Returns:
            Dict[str, Dict[str, Tuple[float, float]]]: _description_
        """
        new_lims: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

        for dim in self.minis.keys():
            max_span = max(self.spans[dim].values())

            for key in self.spans[dim].keys():
                mod = 0.5 * (max_span - self.spans[dim][key])
                new_lims[dim][key] = (self.minis[dim][key] - mod - 0.1 * max_span, self.maxis[dim][key] + mod + 0.1 * max_span)

        return dict(new_lims)

    @staticmethod
    def get_dim(channel: str) -> Literal["VE", "AC", "DS", "FO"]:
        return channel[-4:-2]
