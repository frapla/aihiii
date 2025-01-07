from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from KeyFileGenerator import create_key_file


def create_files(doe: pd.DataFrame, b_path: Path, prefix: str = ""):
    # create new folder
    doe_dir = b_path / f"doe_{prefix}_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}"
    print("Work in", doe_dir)
    doe_dir.mkdir()

    # repeat for all percentiles
    doe = pd.concat([doe] * 3, ignore_index=True)
    doe["PERC"] = np.array([[x] * int(doe.shape[0] / 3) for x in (5, 50, 95)]).flatten()

    # save
    doe.to_excel(doe_dir / "doe.xlsx")

    # create cases
    case_paths: List[Path] = []
    for idx in doe.index:
        body = create_key_file(
            rid=idx,
            percentile=doe.loc[idx, "PERC"],
            pab_t_vent=doe.loc[idx, "PAB_Vent_T"],
            pab_m_scal=doe.loc[idx, "PAB_M_Scal"],
            sll=doe.loc[idx, "SLL"],
            pulse_scale=doe.loc[idx, "Pulse_X_Scale"],
            pulse_angle_deg=doe.loc[idx, "Pulse_Angle"],
            v_init=doe.loc[idx, "V_Init"],
        )
        case_name = body[2]
        run_dir = doe_dir / case_name
        print("Create", run_dir)
        run_dir.mkdir()
        case_paths.append(run_dir / f"{case_name}.key")
        with open(case_paths[-1], "w") as f:
            f.writelines([line + "\n" for line in body])

    bat_body = []
    for case in case_paths:
        bat_body.append(f'echo "Start {case.stem}"')
        bat_body.append(f"cd /d {case.parent.absolute()}")
        bat_body.append(
            f'mpiexec -c 4 -a "C:\Program Files\LSTC\LS-DYNA\R12.1\ls-dyna_mpp_s_R12.1_winx64_ifort170_msmpi.exe" i={case.absolute()} memory=20m > lsrun.out.txt 2>&1'
        )
    bat_body.append('echo "DONE')
    bat_body.append("@pause")
    with open(doe_dir / "run_doe.bat", "w", encoding="cp850") as f:
        f.writelines([line + "\n" for line in bat_body])

    print("Done")
