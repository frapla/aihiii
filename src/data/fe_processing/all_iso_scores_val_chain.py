import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes as Axes

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
from src.utils.Csv import Csv
from src.utils.iso18571 import rating_iso_18571
from src.utils.set_rcparams import set_rcparams
from src.utils.UnifySignal import UnifySignal

LOG: logging.Logger = logging.getLogger(__name__)


READ_NEW: bool = True

B_PATH: Path = Path("/mnt") / "q" / "Val_Chain_Sims" / "AB_Testing"
ISO_PATH: Path = B_PATH / "ISO18571"
ISO_PATH.mkdir(exist_ok=True, parents=True)
REPORT_ASS: Tuple[str, str] = ("HW TH Report", "CAE TH Report")
CASES: Tuple[str, str, str, str, str, str, str, str, str, str, str] = (
    "000_Base_Model",
    "100_Guided_BIW",
    "200_PAB_Simplified",
    "300_Seat_Simplified",
    "400_HIII",
    "500_NoCAB",
    "600_NoDoor",
    "700_Simplified_Belt",
    "800_Simplified_BIW",
    "900_NoIntrusion",
    "950_Dash_Rigid",
    "990_Carpet_Rigid",
)
CONFIGS: Dict[str, str] = {
    "Assemblies_1": "SP 48",
    "Assemblies_2": "DP 48",
    "Assemblies_3": "SP 96",
    "Assemblies_4": "DP 96",
}
UNIFIER: UnifySignal = UnifySignal(
    target_tstart_ms=20,
    target_tend_ms=120,
    target_sampling_rate_ms=0.1,
)
LOAD_CASES: Tuple[str, str, str, str, str] = (
    "Full Frontal",
    "Moderate Overlap Left",
    "Moderate Overlap Right",
    "Oblique Right",
    "Oblique Left",
)
REFERENCES: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)


def read_report_data() -> pd.DataFrame:
    csv = Csv(csv_path=B_PATH / "From_Reports" / "report_data.csv.zip", compress=True)
    db_report: pd.DataFrame = csv.read()

    db_report.replace({"CAE NHTSA": REPORT_ASS[1], "Test NHTSA": REPORT_ASS[0]}, inplace=True)
    db_report.fillna(0, inplace=True)

    return db_report


def read_cae_data() -> List[pd.DataFrame]:
    # read Other Data
    db_caes: List[pd.DataFrame] = []

    for case in CASES:
        LOG.info("Read %s", case)
        csv = Csv(csv_path=B_PATH / case / "extracted.csv.zip", compress=True)

        db_caes.append(csv.read())
        db_caes[-1].replace(CONFIGS, inplace=True)
        db_caes[-1].fillna(0, inplace=True)
        db_caes[-1]["Case"] = [db_caes[-1].loc[idx, "Case"].replace("_", " ") for idx in db_caes[-1].index]

    return db_caes


def label(idx: int, setting: str) -> str:
    return f"{CASES[idx]}_{setting}"


def add_pulse(case: str, db_report: pd.DataFrame, db_caes: List[pd.DataFrame]):
    global REFERENCES
    channels = [
        ["00COG00000VH00VEXD", "00COG00000VH00VEYD"],
        ["00COG00000VH00ACXD", "00COG00000VH00ACYD"],
    ]
    LOG.info("Processing load case %s with %s channels", case, len(sum(channels, [])))

    for ch in sum(channels, []):
        LOG.info("Processing channel %s", ch)
        references = {}
        challengers = {}

        # add report data (always CFC60)
        report = db_report[db_report["Channel"].eq(ch) & db_report["Case"].eq(case)]
        for i, ass in enumerate(REPORT_ASS, 1):
            report2 = report[report["Source"].eq(ass)]
            flip = (
                -1
                if (ch == "00COG00000VH00VEXD" and case != "Full Frontal")
                or (ch == "00COG00000VH00ACXD")
                or (ch == "00COG00000VH00VEYD" and case == "Oblique Left")
                or (ch == "00COG00000VH00ACYD" and case == "Oblique Right")
                else 1
            )

            if report2.shape[0] > 0:
                references[ass] = pd.Series((report2["Value"] * flip).to_numpy(), index=report2["Time"])

        if references:
            references = pd.DataFrame(references)
            LOG.info("References shape before unification %s", references.shape)
            references = UNIFIER.unify(references)
            references.index = [np.round(x, 1) for x in references.index]
            LOG.info("References shape  after unification %s", references.shape)
        else:
            LOG.warning("No report data for channel %s", ch)

        # determine cae flip
        flip = (
            -1
            if (ch == "00COG00000VH00VEXD")
            or (ch == "00COG00000VH00ACXD")
            or (ch == "00COG00000VH00VEYD" and case == "Oblique Left")
            or (ch == "00COG00000VH00VEYD" and case == "Oblique Right")
            or (ch == "00COG00000VH00ACYD" and case == "Oblique Right")
            or (ch == "00COG00000VH00ACYD" and case == "Oblique Left")
            else 1
        )

        # filter CAE data
        for i, db_cae in enumerate(db_caes):
            LOG.info("Processing %s", CASES[i])
            flip2 = 1 if i == 0 else -1
            cae = db_cae[db_cae["Channel"].eq(ch) & db_cae["Case"].eq(case) & (db_cae["Side"].eq("PA") | db_cae["Side"].eq(0))]
            cae.loc[:, "Value"] *= flip * flip2

            # plot single curves
            for setting in sorted(cae["Assembly"].unique()):
                LOG.info("Processing %s", label(i, setting))
                cae2 = cae[cae["Assembly"].eq(setting)].sort_values("Time")
                cae2 = cae2[cae2["Time"].between(UNIFIER.target_tstart_ms, UNIFIER.target_tend_ms)]
                challengers[label(i, setting)] = cae2["Value"].to_numpy()
                time = cae2["Time"].to_numpy()
                LOG.debug("Case %s with shape %s", label(i, setting), challengers[label(i, setting)].shape)

        challengers = pd.DataFrame(challengers, index=time)
        LOG.info("Challengers shape after unification: %s", challengers.shape)

        if isinstance(references, pd.DataFrame):
            references = pd.concat([references, challengers], axis=1)
        else:
            references = challengers.copy()
        LOG.info("References shape after concatenation: %s", references.shape)

        REFERENCES[case][ch] = references.copy()
        LOG.info("Add channel %s of shape %s to references", ch, references.shape)

    LOG.info("References now has %s load cases with %s channels", len(REFERENCES.keys()), len(REFERENCES[case].keys()))


def add_restraint(
    side: Literal[1, 3], case: Literal["Oblique Right", "Oblique Left"], db_report: pd.DataFrame, db_caes: List[pd.DataFrame]
):
    global REFERENCES
    s = {1: "01", 3: "03"}[side]

    channels = [
        [f"{s}FAB00000VH00PRRD", f"{s}BELTBUSLVH00DSRD"],
        [f"{s}BELTB000VH00DSRD", f"{s}BELTB000VH00FORD"],
        [f"{s}BELTB300VH00FORD", f"{s}BELTB400VH00FORD"],
        [f"{s}BELTB500VH00FORD", f"{s}BELTB600VH00FORD"],
    ]
    LOG.info("Processing load case %s side %s with %s channels", case, side, len(sum(channels, [])))

    for ch in sum(channels, []):
        LOG.info("Processing channel %s", ch)
        references = {}
        challengers = {}

        # plot report data
        chh = f"{ch[:10]}TH50{ch[14:-1]}D"
        report = db_report[db_report["Channel"].eq(chh) & db_report["Case"].eq(case)]
        for i, ass in enumerate(REPORT_ASS, 1):
            report2 = report[report["Source"].eq(ass)]
            if report2.shape[0] > 0:
                references[ass] = pd.Series((report2["Value"]).to_numpy(), index=report2["Time"])

        if references:
            references = pd.DataFrame(references)
            LOG.info("References shape before unification %s", references.shape)
            references = UNIFIER.unify(references)
            references.index = [np.round(x, 1) for x in references.index]
            LOG.info("References shape  after unification %s", references.shape)
        else:
            LOG.warning("No report data for channel %s", ch)

        # plot CAE data
        for i, db_cae in enumerate(db_caes):
            LOG.info("Processing %s", CASES[i])
            # filter CAE data
            cae = db_cae[db_cae["Channel"].eq(ch) & db_cae["Case"].eq(case)]
            if "BELTB0" in ch and "DS" in ch:
                cae["Value"] -= cae["Value"].min()

            # plot single curves
            for setting in sorted(cae["Assembly"].unique()):
                cae2 = cae[cae["Assembly"].eq(setting)].sort_values("Time")
                cae2 = cae2[cae2["Time"].between(UNIFIER.target_tstart_ms, UNIFIER.target_tend_ms)]
                challengers[label(i, setting)] = cae2["Value"].to_numpy()
                time = cae2["Time"].to_numpy()
                LOG.debug("Case %s with shape %s", label(i, setting), challengers[label(i, setting)].shape)

        challengers = pd.DataFrame(challengers, index=time)
        LOG.info("Challengers shape after unification: %s", challengers.shape)

        if isinstance(references, pd.DataFrame):
            references = pd.concat([references, challengers], axis=1)
        else:
            references = challengers.copy()
        LOG.info("References shape after concatenation: %s", references.shape)

        REFERENCES[case][ch] = references.copy()
        LOG.info("Add channel %s of shape %s to references", ch, references.shape)

    LOG.info("References now has %s load cases with %s channels", len(REFERENCES.keys()), len(REFERENCES[case].keys()))


def add_body_acc(
    side: Literal[1, 3],
    case: Literal["Oblique Right", "Oblique Left"],
    part: Literal["HEAD", "CHST", "PELV"],
    db_report: pd.DataFrame,
    db_caes: List[pd.DataFrame],
):
    global REFERENCES
    s = {1: "01", 3: "03"}[side]

    channels = [
        [f"{s}{part}0000??50ACRD", f"{s}{part}0000??50ACXD"],
        [f"{s}{part}0000??50ACYD", f"{s}{part}0000??50ACZD"],
    ]
    LOG.info("Processing load case %s side %s with %s channels", case, side, len(sum(channels, [])))

    for ch in sum(channels, []):
        LOG.info("Processing channel %s", ch)
        references = {}
        challengers = {}
        # plot report data
        report = db_report[db_report["Channel"].eq(ch.replace("??", "TH")) & db_report["Case"].eq(case)]
        flip = -1 if "HEAD0000??50ACX" in ch or "HEAD0000??50ACZ" in ch else 1

        for i, ass in enumerate(REPORT_ASS, 1):
            report2 = report[report["Source"].eq(ass)]
            if report2.shape[0] > 0:
                references[ass] = pd.Series((report2["Value"] * flip).to_numpy(), index=report2["Time"])

        if references:
            references = pd.DataFrame(references)
            LOG.info("References shape before unification %s", references.shape)
            references = UNIFIER.unify(references)
            references.index = [np.round(x, 1) for x in references.index]
            LOG.info("References shape  after unification %s", references.shape)
        else:
            LOG.warning("No report data for channel %s", ch)

        # plot CAE data
        for i, db_cae in enumerate(db_caes):
            LOG.info("Add channels %s from setup %s", ch, CASES[i])
            # filter CAE data
            cae = db_cae[db_cae["Channel"].eq(ch.replace("??", "TH" if i < 4 else "H3")) & db_cae["Case"].eq(case)]

            # plot single curves
            for setting in sorted(cae["Assembly"].unique()):
                cae2 = cae[cae["Assembly"].eq(setting)].sort_values("Time")
                cae2 = cae2[cae2["Time"].between(UNIFIER.target_tstart_ms, UNIFIER.target_tend_ms)]
                challengers[label(i, setting)] = cae2["Value"].to_numpy()
                time = cae2["Time"].to_numpy()
                LOG.debug("Case %s with shape %s", label(i, setting), challengers[label(i, setting)].shape)

        challengers = pd.DataFrame(challengers, index=time)
        LOG.info("Challengers shape after unification: %s", challengers.shape)

        if isinstance(references, pd.DataFrame):
            references = pd.concat([references, challengers], axis=1)
        else:
            references = challengers.copy()
        LOG.info("References shape after concatenation: %s", references.shape)

        REFERENCES[case][ch] = references.copy()
        LOG.info("Add channel %s of shape %s to references", ch, references.shape)

    LOG.info("References now has %s load cases with %s channels", len(REFERENCES.keys()), len(REFERENCES[case].keys()))


def add_femur_fo(
    side: Literal[1, 3],
    case: Literal["Oblique Right", "Oblique Left"],
    db_report: pd.DataFrame,
    db_caes: List[pd.DataFrame],
):
    global REFERENCES
    s = {1: "01", 3: "03"}[side]

    channels = [
        [f"{s}FEMRLE00??50FORD", f"{s}FEMRRI00??50FORD"],
    ]

    LOG.info("Processing load case %s side %s with %s channels", case, side, len(sum(channels, [])))

    for ch in sum(channels, []):
        LOG.info("Processing channel %s", ch)
        references = {}
        challengers = {}
        # plot report data
        report = db_report[db_report["Channel"].eq(ch.replace("??", "TH")) & db_report["Case"].eq(case)]
        for i, ass in enumerate(REPORT_ASS, 1):
            report2 = report[report["Source"].eq(ass)]
            if report2.shape[0] > 0:
                references[ass] = pd.Series((report2["Value"]).to_numpy(), index=report2["Time"])

        if references:
            references = pd.DataFrame(references)
            LOG.info("References shape before unification %s", references.shape)
            references = UNIFIER.unify(references)
            references.index = [np.round(x, 1) for x in references.index]
            LOG.info("References shape  after unification %s", references.shape)
        else:
            LOG.warning("No report data for channel %s", ch)

        # plot CAE data
        for i, db_cae in enumerate(db_caes):
            # filter CAE data
            cae = db_cae[db_cae["Channel"].eq(ch.replace("??", "TH" if i < 4 else "H3")) & db_cae["Case"].eq(case)]
            cae.loc[:, "Value"] *= 1 if i < 4 else -1

            # plot single curves
            for setting in sorted(cae["Assembly"].unique()):
                cae2 = cae[cae["Assembly"].eq(setting)].sort_values("Time")
                cae2 = cae2[cae2["Time"].between(UNIFIER.target_tstart_ms, UNIFIER.target_tend_ms)]
                challengers[label(i, setting)] = cae2["Value"].to_numpy()
                time = cae2["Time"].to_numpy()
                LOG.debug("Case %s with shape %s", label(i, setting), challengers[label(i, setting)].shape)

        challengers = pd.DataFrame(challengers, index=time)
        LOG.info("Challengers shape after unification: %s", challengers.shape)

        if isinstance(references, pd.DataFrame):
            references = pd.concat([references, challengers], axis=1)
        else:
            references = challengers.copy()
        LOG.info("References shape after concatenation: %s", references.shape)

        REFERENCES[case][ch] = references.copy()
        LOG.info("Add channel %s of shape %s to references", ch, references.shape)

    LOG.info("References now has %s load cases with %s channels", len(REFERENCES.keys()), len(REFERENCES[case].keys()))


def calculate_iso_scores(refs: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed=42)
    debug_signals = ISO_PATH / "debug_signals"
    debug_signals.mkdir(exist_ok=True, parents=True)
    all_isos: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
    for case in refs.keys():
        for channel in refs[case].keys():
            LOG.info(
                "Processing %s %s with %s references and %s challengers",
                case,
                channel,
                refs[case][channel].shape[1],
                refs[case][channel].shape[1],
            )
            iso_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
            for chal in refs[case][channel].columns:
                for ref in refs[case][channel].columns:
                    if chal in iso_ratings and ref in iso_ratings[chal]:
                        # mirror symmetric result
                        LOG.info("Add symmetric result %s to %s", ref, chal)
                        iso_ratings[ref][chal] = iso_ratings[chal][ref]
                        LOG.debug(
                            "ISO of %s vs %s for channel %s in case %s is %s", ref, chal, channel, case, iso_ratings[chal][ref]
                        )
                    elif ("Report" in chal or "Report" in ref) or chal[-5:] == ref[-5:]:
                        # calculate ISO
                        LOG.info("For case %s, channel %s: Get ISO of %s vs %s", case, channel, ref, chal)
                        iso_ratings[chal][ref] = rating_iso_18571(
                            signal_ref=refs[case][channel][ref].to_numpy(),
                            signal_comp=refs[case][channel][chal].to_numpy(),
                        )["ISO 18571 Rating"]
                        LOG.debug(
                            "ISO of %s vs %s for channel %s in case %s is %s", ref, chal, channel, case, iso_ratings[chal][ref]
                        )

                        # debug plot
                        if rng.random() < 0.8:
                            fig, ax = plt.subplots()
                            ax.plot(refs[case][channel].index, refs[case][channel][ref], label=f"Reference: {ref}")
                            ax.plot(
                                refs[case][channel].index,
                                refs[case][channel][chal],
                                label=f"Challenger: {chal}",
                                ls="--",
                            )
                            ax.grid()
                            ax.set_xlabel("Time [ms]")
                            ax.set_ylabel(channel)
                            ax.set_title(f"{case} - ISO18571={iso_ratings[chal][ref]:.3f}")
                            ax.legend()
                            ax.axhline(0, c="black")
                            f_name = debug_signals / f"{case}_{channel.replace('??', 'DM')}_{ref}_{chal}.jpg"
                            fig.savefig(fname=f_name)
                            plt.close(fig)
                            LOG.debug("Saved debug plot to %s", f_name)
                    else:
                        # skip
                        LOG.warning("Skip %s vs %s", ref, chal)

            # store
            all_isos[case][channel] = pd.DataFrame(iso_ratings)
            Csv(
                csv_path=ISO_PATH / f"{case.replace(' ', '_')}_{channel.replace(' ', '_').replace('??', 'DM')}",
                compress=True,
            ).write(all_isos[case][channel])
            LOG.info("ISOs for %s %s with shape %s", case, channel, all_isos[case][channel].shape)
    return all_isos


def isolate(names: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)

    for name in names:
        if name.endswith("Report"):
            groups[name].append(name)
        else:
            groups[name[:-6]].append(name)

    return dict(groups)


def get_grouped_isos(iso_scores: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    all_grouped_isos: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

    for case in iso_scores.keys():
        for channel in iso_scores[case].keys():
            LOG.info("Processing %s %s", case, channel)
            db: pd.DataFrame = iso_scores[case][channel].copy()
            db.rename(
                columns={x: x.replace("_", " ").replace("CAE ", "") for x in db.columns if "Report" not in x},
                index={x: x.replace("_", " ").replace("CAE ", "") for x in db.index if "Report" not in x},
                inplace=True,
            )

            ref_groups = isolate(db.index)
            chal_groups = isolate(db.columns)

            iso_avgs = defaultdict(dict)
            for ref_group in ref_groups.keys():
                for chal_group in chal_groups.keys():
                    selection = db.loc[ref_groups[ref_group], chal_groups[chal_group]]
                    LOG.debug("Selection of %s to %s has shape %s", ref_group, chal_group, selection.shape)
                    iso_avgs[ref_group][chal_group] = selection.median(axis=None)
            all_grouped_isos[case][channel] = pd.DataFrame(iso_avgs)
            LOG.info("Grouped ISOs for %s %s with shape %s", case, channel, all_grouped_isos[case][channel].shape)

    return dict(all_grouped_isos)


def show_isos(
    isos_grouped: Dict[str, Dict[str, pd.DataFrame]],
    case: Literal["Full Frontal", "Oblique Left", "Oblique Right"],
    channels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    cmap: Optional[str] = None,
    norm: Optional[str] = None,
) -> None:
    refs = ["HW TH Report", "CAE TH Report"] + [x.replace("_", " ") for x in CASES]
    chals = refs.copy()

    for channel in isos_grouped[case].keys() if channels is None else channels:
        LOG.info("Processing %s %s", case, channel)

        av_refs = set(isos_grouped[case][channel].columns)
        av_chals = set(isos_grouped[case][channel].index)

        selected = defaultdict(dict)
        for ref in refs:
            for chal in chals:
                if ref in av_refs and chal in av_chals:
                    if ref == chal:
                        selected[ref][chal] = np.nan
                    else:
                        selected[ref][chal] = isos_grouped[case][channel].loc[chal, ref]

                else:
                    selected[ref][chal] = np.nan
        to_plot = pd.DataFrame(selected, index=chals)
        to_plot.rename(
            index={idx: " ".join(idx.split()[1:]) for idx in to_plot.index if "Report" not in idx},
            columns={idx: " ".join(idx.split()[1:]) for idx in to_plot.columns if "Report" not in idx},
            inplace=True,
        )

        if ax is None:
            _, ax = plt.subplots(layout="constrained")
        sns.heatmap(
            to_plot[list(to_plot.index[:-1])],
            annot=True,
            mask=np.triu(np.ones_like(to_plot[list(to_plot.index[:-1])], dtype=bool), k=1),
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="magma" if cmap is None else cmap,
            linewidth=0.7,
            norm=norm,
            fmt=".2f",
            annot_kws={"fontsize": 4},
        )
        ax.set_title(channel)
        ax.set_xlabel("Reference")
        ax.set_ylabel("Comparison")
        ax.grid()
        ax.set_axisbelow(True)


def plot_channel_group(
    isos_grouped: Dict[str, Dict[str, pd.DataFrame]],
    case: str,
    channels: List[List[str]],
    formats: Optional[List[str]] = None,
    grp_name: Optional[str] = None,
) -> None:
    LOG.info("Processing load case %s with %s channels", case, len(sum(channels, [])))

    fig_width: float = 1 * (448.13095 / 72)
    fig_height: float = 1.1 * fig_width
    fig, ax = plt.subplot_mosaic(
        [["none"] * len(channels[0]), *channels],
        figsize=(fig_width, 0.5 * len(channels) * fig_height),
        layout="constrained",
        gridspec_kw={"height_ratios": (0.025, *([1] * len(channels)))},
    )
    # fig.suptitle(f"{case} - ISO 18571 Rating")
    cmap = mpl.colors.ListedColormap(["indianred", "orange", "yellowgreen", "forestgreen"])
    bounds = [0, 0.58, 0.8, 0.94, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax["none"],
        location="top",
    )
    ax["none"].set(frame_on=False)
    ax["none"].set_xticklabels([f"{x:.2f}" for x in bounds])

    for ch in sum(channels, []):
        LOG.info("Processing channel %s", ch)
        show_isos(isos_grouped=isos_grouped, case=case, channels=[ch], ax=ax[ch], cmap=cmap, norm=norm)

    p_path = ISO_PATH / "Figures"
    p_path.mkdir(exist_ok=True, parents=True)
    if formats is not None:
        for fmt in formats:
            fp_path = p_path / fmt.upper()
            fp_path.mkdir(exist_ok=True)
            plt.savefig(fp_path / f"{case.replace(' ', '_')}_{grp_name}.{fmt}")
        plt.close(fig)


def plot_pile(isos_grouped: Dict[str, Dict[str, pd.DataFrame]]):
    # pulse
    channel_sets = {
        "Pulses": [
            ["00COG00000VH00VEXD", "00COG00000VH00VEYD"],
            ["00COG00000VH00ACXD", "00COG00000VH00ACYD"],
        ]
    }

    # RHS
    for s in ("03",):
        # RHS
        channel_sets[f"{s}_RHS"] = [
            [f"{s}FAB00000VH00PRRD", f"{s}BELTBUSLVH00DSRD"],
            [f"{s}BELTB000VH00DSRD", f"{s}BELTB000VH00FORD"],
            [f"{s}BELTB300VH00FORD", f"{s}BELTB400VH00FORD"],
            [f"{s}BELTB500VH00FORD", f"{s}BELTB600VH00FORD"],
        ]

        # body
        for part in ("HEAD", "CHST", "PELV"):
            channel_sets[f"{s}_{part}"] = [
                [f"{s}{part}0000??50ACRD", f"{s}{part}0000??50ACXD"],
                [f"{s}{part}0000??50ACYD", f"{s}{part}0000??50ACZD"],
            ]

        # femur
        channel_sets[f"{s}_FMR"] = [
            [f"{s}FEMRLE00??50FORD", f"{s}FEMRRI00??50FORD"],
        ]

    for case in LOAD_CASES:
        for channel_set in channel_sets.keys():
            plot_channel_group(
                isos_grouped=isos_grouped,
                case=case,
                channels=channel_sets[channel_set],
                formats=["png", "pdf"],
                grp_name=channel_set,
            )


def main():
    set_rcparams()
    # get data
    LOG.info("Read data")
    db_report = read_report_data()
    db_caes = read_cae_data()
    LOG.info("Data reading done")

    # collect combinations
    LOG.info("Collect combinations")

    for cs in LOAD_CASES:
        LOG.info("Processing load case %s", cs)
        add_pulse(case=cs, db_report=db_report, db_caes=db_caes)
        add_restraint(case=cs, side=3, db_report=db_report, db_caes=db_caes)
        add_femur_fo(case=cs, side=3, db_report=db_report, db_caes=db_caes)

        for part in ("HEAD", "CHST", "PELV"):
            add_body_acc(case=cs, side=3, part=part, db_report=db_report, db_caes=db_caes)
    LOG.info("Combinations collected")

    LOG.info("Calculate ISO scores")
    isos = calculate_iso_scores(refs=REFERENCES)
    isos_groups = get_grouped_isos(iso_scores=isos)
    LOG.info("ISO scores calculated")

    LOG.info("Plotting")
    plot_pile(isos_grouped=isos_groups)
    LOG.info("Plotting done")


if __name__ == "__main__":
    custom_log.init_logger(logging.INFO)
    LOG.info("Start processing")
    main()
    LOG.info("Processing finished")
