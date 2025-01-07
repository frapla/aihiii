from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as sconst


def get_olc(
    channel: pd.Series,
    free_movement: float = 0.065,
    decal_way: float = 0.235,
    scal_x2s: float = 1.0,
    scal_y2mps: float = 1.0,
    plot: bool = False,
) -> Tuple[float, float, float]:
    """Calculate approximate OLC in g (accuracy determined by sampling rate)

    Args:
        channel (pd.Series): the signal, values are signal values, index is time
        free_movement (float, optional): Free motion of occupant relative to vehicle. Defaults to 0.065.
        decal_way (float, optional): Restrained motion of occupant relative to vehicle. Defaults to 0.235.
        scal_x2s (float, optional): time scaling factor to s. Defaults to 1.0.
        scal_y2mps (float, optional): velocity scaling factor to m/s. Defaults to 1.0.
        plot (bool, optional): debug plots. Defaults to False.

    Returns:
        Tuple[float, float, float]: _description_: OCL in g, t start deceleration in s, t end deceleration in s
    """

    # prepare channel
    ch_scaled = __prepare_channel(channel, scal_x2s, scal_y2mps)

    # calculate t1 (area under velocity = free_movement)
    t1, search_pnt_t1 = __calculate_t1(ch_scaled, free_movement, plot)

    # calculate t2 (area between virtual occupant movement and car = decal_way)
    t2, search_pnt_t2 = __calculate_t2(ch_scaled, t1, decal_way, plot)

    # calculate OLC
    olc = __calculate_olc(ch_scaled, t1, t2)

    if plot:
        __make_plot(ch_scaled, t1, t2, olc, search_pnt_t1, search_pnt_t2)

    return olc, t1, t2


def __prepare_channel(channel: pd.Series, scal_x: float, scal_y=float) -> pd.Series:
    ch_scaled = pd.Series(channel.values * scal_y, index=channel.index * scal_x, dtype=float)
    v_min = ch_scaled.min()
    if v_min < 0:
        ch_scaled += abs(ch_scaled.min())

    return ch_scaled


def __calculate_t1(ch_scaled: pd.Series, free_movement: float, plot: bool) -> Tuple[float, Optional[pd.Series]]:
    v0 = ch_scaled.loc[0]

    search_pnt_t1 = pd.Series([v0], index=[0]) if plot else None

    for t1 in ch_scaled.index:
        x_veh_free = np.trapz(y=ch_scaled.loc[0:t1].values, x=ch_scaled.loc[0:t1].index)
        x_occ_global_free = v0 * t1
        x_occ_local_free = abs(x_veh_free - x_occ_global_free)
        v1 = ch_scaled.loc[t1]
        if plot:
            search_pnt_t1.at[t1] = v1
        if x_occ_local_free >= free_movement:
            break

    return t1, search_pnt_t1


def __calculate_t2(ch_scaled: pd.Series, t1: float, decal_way: float, plot: bool) -> Tuple[float, Optional[pd.Series]]:
    v0 = ch_scaled.loc[0]

    search_pnt_t2 = pd.Series([ch_scaled.loc[t1]], index=[t1]) if plot else None

    for t2 in ch_scaled.loc[t1:].index:
        x_veh_restr = np.trapz(y=ch_scaled.loc[t1:t2].values, x=ch_scaled.loc[t1:t2].index)
        v2 = ch_scaled.loc[t2]
        x_occ_global_restr = (0.5 * v0 + v2) * (t2 - t1)
        x_occ_local_restr = abs(x_veh_restr - x_occ_global_restr)

        if plot:
            search_pnt_t2.at[t2] = v2
        if x_occ_local_restr >= decal_way:
            break

    return t2, search_pnt_t2


def __calculate_olc(ch_scaled: pd.Series, t1: float, t2: float) -> float:
    v0 = ch_scaled.loc[0]
    olc = abs(((ch_scaled.loc[t2] - v0) / (t2 - t1))) / sconst.g

    return olc


def __make_plot(ch_scaled, t1, t2, olc, search_pnt_t1, search_pnt_t2):
    v0 = ch_scaled.loc[0]
    v1 = ch_scaled.loc[t1]
    v2 = ch_scaled.loc[t2]

    _, ax = plt.subplots()
    ax.plot(ch_scaled, color="blue")
    ax.plot([0, t1], [v0, v0], color="red")
    ax.plot([t1, t1], [0, v0], linestyle=":", color="black")
    ax.plot([t2, t2], [0, v2], linestyle=":", color="black")
    ax.plot([t1, t2], [v0, v2], color="red")
    ax.plot([t2, max(ch_scaled.index)], [v2, v2], color="red")
    ax.plot(search_pnt_t1, marker="o", color="orange", markersize=0.6)
    ax.plot(search_pnt_t2, marker="o", color="purple", markersize=0.6)
    ax.xlabel("time [s]")
    ax.ylabel("velocity [m/s]")
    ax.annotate("t1 {0:.3f}s".format(t1), (t1 + 0.001, 1))
    ax.annotate("t2 {0:.3f}s".format(t2), (t2 + 0.001, 1))
    ax.annotate("v1 {0:.3f}m/s".format(v1), (t1, v1 + 0.6))
    ax.annotate("v2 {0:.3f}m/s".format(v2), (t2, v2 + 0.6))
    ax.annotate("OLC {0:.2f}g".format(olc), ((t2 - t1) / 2, v2 / 2))
    ax.grid(True)
    ax.show()
