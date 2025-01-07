from typing import List

import numpy as np


def create_key_file(
    rid: int,
    percentile: int,
    pab_t_vent: float,
    pab_m_scal: float,
    sll: float,
    pulse_scale: float = 1,
    pulse_angle_deg: float = 0,
    v_init: float = -15560.0,
) -> List[str]:
    perc = f"{percentile:02}"
    c_name = f"V{rid:07}"
    c_name = c_name.replace(".", "d")
    pulse_angle_rad = np.radians(pulse_angle_deg)
    pulse_angle_rad = 1e-10 if pulse_angle_rad == 0 else pulse_angle_rad

    body = [
        "*KEYWORD",
        "*TITLE",
        c_name,
        "*PARAMETER",
        "$#   prmr1      val1     prmr2      val2     prmr3      val3     prmr4      val4",
        "$ Initial velocity of vehicle [mm/s]",
        f"R INI_VEL {v_init:+.3e}",
        "$ time to open adaptive vent [s]",
        f"R PABTVENT{pab_t_vent:+.3e}",
        "$ Scaling factor for proportional scaling of PAB inflator mass flow [1]",
        f"R PABPSCAL{pab_m_scal:+.3e}",
        "$ Retractor Load Limiter B0 (B3~1.9*B0) [N]",
        f"R SLL     {sll:+.3e}",
        "$ Scaling of x acceleration of vehicle [1], Rotation angle of pulse [rad]",
        f"R PSCAL   {pulse_scale:+.3e}",
        f"R ALPHA   {pulse_angle_rad:+.3e}",
        "*INCLUDE",
        "../../../Pool/DATABASE_CONTROL.k",
        "../../../Pool/MATERIALS.k",
        "../../../Pool/MOTION_SIMPLE.k",
        "../../../Pool/DATA_NHTSA_FULL_FRONTAL_COG_PARA.k",
        "../../../Pool/BIW.k",
        "../../../Pool/DASHBOARD.k",
        "../../../Pool/CARPET_PA.k",
        "../../../Pool/FAB_PA_withInner_withLiner_infl60_coarse_shrinked.k",
        f"../../../Pool/SEAT_PA_{perc}th_HIII_ml_br19_sr17.k",
        f"../../../Pool/BELT_PA_{perc}th_HIII_ml_br19_sr17.k",
        f"../../../Pool/DUMMY_{perc}th_HIII_FAST_PA_ml_br19_sr17.k",
        "../../../Pool/INTER_CONTACTS.k",
        "*END",
    ]

    return body
