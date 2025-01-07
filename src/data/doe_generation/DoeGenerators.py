import pandas as pd
import numpy as np
from itertools import product


def create_doe_full_fact_with_jitter(levels: int, rand_strengh: float) -> pd.DataFrame:
    # factors
    factors = {"PABTVENT": [0.85, 1.2], "PABPSCAL": [1.0, 1.5], "SLL": [2300, 3000]}
    facts = sorted(factors.keys())

    # factor levels
    sf = {k: np.linspace(factors[k][0], factors[k][1], levels) for k in facts}

    # combine all with all
    doe = pd.DataFrame(product(*[sf[k] for k in facts]), columns=facts)

    # randomize
    rands = np.random.rand(*doe.shape) - 0.5
    rands *= np.array([[rand_strengh * (sf[k][1] - sf[k][0])] for k in facts]).T
    doe += rands

    # restore factor ranges
    orig_ranges = pd.Series({k: factors[k][-1] - factors[k][0] for k in facts})
    cur_ranges = doe.max() - doe.min()
    doe -= doe.min()
    doe *= orig_ranges / cur_ranges
    doe += pd.Series({k: factors[k][0] for k in facts})

    return doe


def create_3lvl_ab():
    f_devs = {
        "V_Init": (-15560, 3000),
        "Pulse_X_Scale": (1, 0.2),
        "Pulse_Angle": [0, 10],
        "PAB_M_Scal": [1, 0.1],
        "PAB_Vent_T": [0.1, 0.03],
        "SLL": [2340, 200],
    }
    f_names = sorted(f_devs.keys())

    f_levels = []
    for f_name in f_names:
        for i in [-1, 0, 1]:
            f_level = []
            for f_name_ in f_names:
                if f_name != f_name_:
                    f_level.append(f_devs[f_name_][0])
                else:
                    f_level.append(f_devs[f_name_][0] + i * f_devs[f_name_][1])
            f_levels.append(f_level)

    f_levels = pd.DataFrame(f_levels, columns=f_names).drop_duplicates(
        ignore_index=True
    )

    return f_levels


def create_doe_full_fact(n_lvls=5):
    n_lvls = n_lvls  # should be odd number

    f_devs = {
        "V_Init": (-15560, 3000),
        "Pulse_X_Scale": (1, 0.2),
        "Pulse_Angle": [0, 10],
        "PAB_M_Scal": [1, 0.1],
        "PAB_Vent_T": [0.1, 0.03],
        "SLL": [2340, 200],
    }
    f_names = sorted(f_devs.keys())

    f_ranges = {
        k: np.linspace(f_devs[k][0] - f_devs[k][1], f_devs[k][0] + f_devs[k][1], n_lvls)
        for k in f_names
    }
    f_levels = (
        np.array(np.meshgrid(*[f_ranges[k] for k in f_names]))
        .reshape(len(f_devs.keys()), -1)
        .T
    )
    f_levels = pd.DataFrame(f_levels, columns=f_names).drop_duplicates()

    return f_levels
