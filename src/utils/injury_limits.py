import pandas as pd
from typing import List, Optional


def get_limits_biomechanical() -> pd.DataFrame:
    """Biomechanical Limits

    Returns:
        pd.DataFrame: limits with percentiles as index and criteria as columns
    """
    bio_lim = {
        "Chest_Deflection": [52, 63, 70],  # FMVSS208
        "Chest_a3ms": 60,  # FMVSS208
        "Femur_Fz_Max": [6.805, 10, 12.7],  # FMVSS208
        "Head_HIC15": 700,  # FMVSS208
        "Head_HIC36": 1000,  # UN R137
        "Head_a3ms": 80,  # UN R137
        "Neck_Nij": 1,  # FMVSS208
    }

    bio_lim_db = pd.DataFrame(index=(5, 50, 95))
    for lim in sorted(bio_lim.keys()):
        bio_lim_db[lim] = bio_lim[lim]
    bio_lim_db.index.name = "PERC"

    return bio_lim_db


def get_limits_euro_ncap(buffers: Optional[List[float]] = None) -> pd.DataFrame:
    """0 and 4 Points from Euro NCAP 2024 High Speed HIII 5th and 50th percentile
    95th percentile is extrapolated from 5th and 50th
    Points thresholds are linear interpolated between 0 and 4

    Returns:
        pd.DataFrame: Limits with multindex (POINTS, PERC), columns are criteria
    """
    # from Euro NCAP HIII 5th and 50th percentile, upper and lower bounds
    all_lims = {
        "Head_HIC15": [[500, 500], [700, 700]],  # [[lo 5, lo 50], [up 5, up 50]]
        "Head_a3ms": [[72, 72], [80, 80]],
        "Neck_My_Extension": [[36, 42], [49, 57]],
        "Neck_Fz_Max_Tension": [[1.7, 2.7], [2.62, 3.3]],
        "Neck_Fx_Shear_Max": [[1.2, 1.9], [1.95, 3.1]],
        "Chest_Deflection": [[18, 22], [34, 42]],
        "Femur_Fz_Max_Compression": [[2.6, 3.8], [6.2, 9.07]],
        "Chest_VC": [[0.5, 0.5], [1.0, 1.0]],
    }

    # add 95th
    for thres in all_lims.keys():
        for i, bound in enumerate(all_lims[thres]):
            all_lims[thres][i].append(2 * bound[1] - bound[0])

    # reformat
    all_lims = pd.DataFrame(all_lims, index=(4, 1))
    all_lims["PERC"] = [[5, 50, 95]] * all_lims.shape[0]
    all_lims.index.name = "POINTS"
    all_lims = all_lims.explode(list(all_lims.columns))
    all_lims.set_index("PERC", append=True, inplace=True)

    # interpolate
    for perc in all_lims.index.get_level_values("PERC").unique():
        y1 = all_lims.loc[(1, perc), :].values
        y4 = all_lims.loc[(4, perc), :].values

        n = y1 - (y4 - y1) / (4 - 1)
        m = (y4 - y1) / (4 - 1)

        for i in range(2, 4):
            all_lims.loc[(i, perc), :] = m * i + n

        if buffers is not None:
            for buffer in buffers:
                all_lims.loc[(1 - buffer, perc), :] = all_lims.loc[(1, perc), :] * (1 + buffer)
                all_lims.loc[(4 + buffer, perc), :] = all_lims.loc[(4, perc), :] * (1 - buffer)

    all_lims.sort_index(inplace=True)

    return all_lims


if __name__ == "__main__":
    print("Biomechanical limits:")
    print(get_limits_biomechanical())
    print("\nEuro NCAP limits:")
    print(get_limits_euro_ncap(buffers=[0.2]))
