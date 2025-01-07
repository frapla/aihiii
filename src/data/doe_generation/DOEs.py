from pathlib import Path

from CreateFiles import create_files
from DoeGenerators import create_3lvl_ab, create_doe_full_fact

create_files(
    doe=create_3lvl_ab(),
    b_path=Path(r"Q:\Honda_Accord_2014_Sled_with_HIII_Rigid_Seat_SpeedOpt_BigDOE"),
    prefix="para_infl",
)

create_files(
    doe=create_doe_full_fact(),
    b_path=Path(r"Q:\Honda_Accord_2014_Sled_with_HIII_Rigid_Seat_SpeedOpt_BigDOE"),
    prefix="big_grid",
)
