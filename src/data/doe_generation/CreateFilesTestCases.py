import argparse
import sys
import logging
from pathlib import Path
from typing import List
import numpy as np

import pandas as pd
import tqdm

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log

LOG: logging.Logger = logging.getLogger(__name__)


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
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$ PARAMETER",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
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
        "*PARAMETER",
        "$ TIME TO SUB FRAME FAILURE BASED ON THE LOAD CASE",
        "$ 01_US_NCAP_Full_Frontal = 40.4 ms",
        "R TFAIL           0.",
        "R CAW             0.",
        "R CAF             0.",
        "R CAR             0.",
        "$",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$ OUTPUT",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$",
        "*DATABASE_EXTENT_BINARY",
        "$#   NEIPH     NEIPS    MAXINT    STRFLG    SIGFLG    EPSFLG    RLTFLG    ENGFLG",
        "         0         0         0         0         2         1         2         1",
        "$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat",
        "         0         1         0         1         1         3         2         1",
        "$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout",
        "                   0       1.0         0         0         0",
        "$#    dtdt    resplt     neipb   quadsld    cubsld   deleres",
        "         0",
        "*DATABASE_BINARY_D3PLOT",
        "$#      dt      lcdt      beam     npltc    psetid",
        "     0.035",
        "$#   ioopt      rate    cutoff    window      type      pset",
        "         0",
        "*DATABASE_SBTOUT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_MATSUM",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_GLSTAT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_SLEOUT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_RCFORC",
        "$#      dt    binary      lcur     ioopt",
        "  0.000010         2         0         1",
        "*DATABASE_RWFORC",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_DEFORC",
        "$#      dt    binary      lcur     ioopt",
        "  0.001000         2         0         1",
        "*DATABASE_ABSTAT",
        "$#      dt    binary      lcur     ioopt",
        "  0.001000         2         0         1",
        "*DATABASE_JNTFORC",
        "$#      dt    binary      lcur     ioopt",
        "  0.001000         2         0         1",
        "*DATABASE_NODOUT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_SECFORC",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_ELOUT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "*DATABASE_SSSTAT",
        "$#      dt    binary      lcur     ioopt",
        "  0.000100         2         0         1",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$ INCLUDE PATH                                                                 $",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$",
        "$	00	$	DUMMY",
        "$____________________________________",
        "*PARAMETER",
        "$: friction dummy-skin <--> Airbag",
        "R FS_D_AB0      1.00",
        "R FD_D_AB0      1.00",
        "$: friction dummy-leg <--> DAB",
        "R FS_D_AB1      0.20",
        "R FD_D_AB1      0.20",
        "$: friction dummy-thorax <--> DAB",
        "R FS_D_AB2      0.40",
        "R FD_D_AB2      0.40",
        "$: friction dummy <--> seat",
        "R dums          0.35",
        "R dumd          0.35",
        "$: friction dummy <--> carpet",
        "R dfrs          1.0",
        "R dfrd          1.0",
        "$: friction dummy <--> IP",
        "R dips          0.4",
        "R dipd          0.4",
        "$: friction belt <--> dummy & seat",
        "R dsbs          0.15",
        "R dsbd          0.15",
        "$",
        "*INCLUDE",
        "$#                                                                      filename",
        f"../../00_INCLUDES/00_DUMMY/DUMMY_{perc}th_HIII_FAST_PA_ml_br19_sr17.k",
        "$",
        "$	01	$	AIRBAG",
        "$____________________________________",
        "*PARAMETER",
        "$: time to fire driver airbag",
        "R TTF_DAB      0.014",
        "R TTF_CAB      0.042",
        "R TTF_PAB      0.016",
        "$ -----------------------------------------------------------------------",
        "*INCLUDE",
        "../../00_INCLUDES/67_PAB/FAB_PA_withInner_withLiner_infl60_coarse_shrinked.k",
        "$ -----------------------------------------------------------------------",
        "$ : DRIVER SIDE CURTAIN AIRBAG",
        "*PARAMETER",
        "$ CAB deploy time - changes contact/mass flowrate birth/death time",
        "R DPLY_T       0.042",
        "*PARAMETER_EXPRESSION",
        "R DPLY_TMSDPLY_T*1000",
        "R DPLY_TOFDPLY_T-0.001",
        "*DEFINE_TRANSFORMATION_TITLE",
        "CAB_POSITION",
        "  71000002",
        "TRANSL            0.       35.        0.",
        "*INCLUDE",
        "../../00_INCLUDES/71_CAB_DR/CAB_L_V013.k",
        "*NODE_TRANSFORM",
        "  71000001  71000101",
        "*DEFINE_TRANSFORMATION_TITLE",
        "CURNTAIN_AIRBAG_SYMMETRY",
        "  71000001",
        "SCALE             1.       -1.        1.",
        "*INCLUDE",
        "../../00_INCLUDES/71_CAB_DR/CAB_R_V013_CONNECTIONS_mod.k",
        "*INCLUDE",
        "../../00_INCLUDES/71_CAB_DR/71_CAB_CONTACT.k",
        "$ -----------------------------------------------------------------------",
        "$ -----------------------------------------------------------------------",
        "$ : DRIVER_SIDE_SEAT_BELT",
        "*PARAMETER",
        "$: time to fire pretensioner",
        "R TTF_Dpb       13.0",
        "$: time to fire load limiter driver",
        "R TTF_DLL       0.04",
        "$",
        "$: load limiter step 1 driver",
        "R LL_DS1         3.2",
        "$: load limiter step 2 driver",
        "R LL_DS2         2.3",
        "$: friction d-ring",
        "R  FDRING       0.35",
        "R  FSDRING      0.35",
        "$: friction belt vs. buckle",
        "R  FDBUCK       1.00",
        "R  FSBUCK       0.50",
        "R  DC         0.0001",
        "*INCLUDE",
        f"../../00_INCLUDES/65_SEATBELT/04_belt_pa_030_mod_{perc}th.k",
        "$",
        "$	46	$	CARPET",
        "$____________________________________",
        "*INCLUDE",
        "../../00_INCLUDES/46_CARPETS/carpets_V01_mod.k",
        "$",
        "$	02	$	SUSPENSION_FR",
        "$____________________________________",
        "$	10	$	BIW",
        "$____________________________________",
        "*INCLUDE",
        "../../00_INCLUDES/10_BIW/BIW_8_mod.k",
        "$",
        "$____________________________________",
        "$	15	$	IP",
        "$____________________________________",
        "*INCLUDE",
        "../../00_INCLUDES/15_IP/IP_6_mod.k",
        "$",
        "$	16	$	DASH",
        "$____________________________________",
        "*INCLUDE",
        "../../00_INCLUDES/16_DASH/Dashboard_6_mod.k",
        "$",
        "$",
        "$	42	$	DOOR_FR",
        "$____________________________________",
        "*INCLUDE",
        "../../00_INCLUDES/41_DOOR_FR/41_DOOR_FR_3_mod.k",
        "$",
        "*INCLUDE",
        f"../../00_INCLUDES/60_SEATS/SEAT_PA_{perc}th_HIII_ml_br19_sr17.k",
        "$",
        "$-------------------------------------------------------------------------------",
        "$	80	$	MATERIALS",
        "$-------------------------------------------------------------------------------",
        "$",
        "*INCLUDE",
        "../../00_INCLUDES/80_MATERIALS/Materials_DB_Accord_004.k",
        "$",
        "$-------------------------------------------------------------------------------",
        "$	81	$	CONNECTIONS",
        "$-------------------------------------------------------------------------------",
        "$",
        "*INCLUDE",
        "../../00_INCLUDES/81_CONNECTIONS/CONNECTIONS_5_mod_3.k",
        "$",
        "$",
        "$-------------------------------------------------------------------------------",
        "$	85	$	Mass",
        "$-------------------------------------------------------------------------------",
        "*INCLUDE",
        "../../00_INCLUDES/85_MASS/85_MASS_V002_mod.k",
        "$",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$	90	$	BARRIERS AND LOAD CASE",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$",
        "*INCLUDE",
        "../../00_INCLUDES/99_MOTION/MOTION_SIMPLE.k",
        "../../00_INCLUDES/99_MOTION/DATA_NHTSA_FULL_FRONTAL_COG_PARA.k",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$	92	$	CONTACT",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "*INCLUDE",
        "../../00_INCLUDES/70_CONTACTS/92_Contact_12_06302017_V01_mod_4.k",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$	98	$	CONTROL",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "*INCLUDE",
        "../../00_INCLUDES/98_CONTROL_CARDS/98_Control_card_003_05302017.k",
        "$---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8",
        "$-------------------------------------------------------------------------------",
        "$ GRAVITY",
        "$-------------------------------------------------------------------------------",
        "*LOAD_BODY_Z",
        "$#    LCID        SF    LCIDDR        XC        YC        ZC       CID",
        "   1000001 9806.0000",
        "*DEFINE_CURVE",
        "$#    LCID      SIDR       SFA       SFO      OFFA      OFFO    DATTYP",
        "   1000001         0  1.000000  1.000000",
        "$#                A1                  O1",
        "               0.000           1.0000000",
        "           5.0000000           1.0000000",
        "$-------------------------------------------------------------------------------",
        "$ GLOBAL CONTACT",
        "$-------------------------------------------------------------------------------",
        "*CONTACT_AUTOMATIC_SINGLE_SURFACE_ID",
        "$#     CID                                                                 TITLE",
        "      1000GLOBAL_VEHICLE_CONTACT",
        "$#    SSID      MSID     SSTYP     MSTYP    SBOXID    MBOXID       SPR       MPR",
        "  70000001         0         6         0         0         0         0         0",
        "$#      FS        FD        DC        VC       VDC    PENCHK        BT        DT",
        "       0.2       0.1     0.001       0.0       0.0         0       0.01.00000E20",
        "$#     SFS       SFM       SST       MST      SFST      SFMT       FSF       VSF",
        "       0.0       1.0       0.0       0.0       1.0       1.0       1.0       1.0",
        "$#    SOFT    SOFSCL    LCIDAB    MAXPAR     SBOPT     DEPTH     BSORT    FRCFRQ",
        "         1       0.1         0     1.025       2.0         2         0         1",
        "$#  PENMAX    THKOPT    SHLTHK     SNLOG      ISYM     I2D3D    SLDTHK    SLDSTF",
        "       0.0         0         0         0         0         0       0.0       0.0",
        "$#    IGAP    IGNODPRFAC/MPADTSTIF/MPAR2   UNUSED     UNUSED    FLANGL   CID_RCF",
        "         0         1       0.0       0.0                           0.0         0",
        "$",
        "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_ID",
        "$#     CID                                                                 TITLE",
        "      1001GLOBAL_VEHICLE_CONTACT_SYM",
        "$#    SSID      MSID     SSTYP     MSTYP    SBOXID    MBOXID       SPR       MPR",
        "  70000001  10000749         6         2         0         0         0         0",
        "$#      FS        FD        DC        VC       VDC    PENCHK        BT        DT",
        "       0.2       0.1     0.001       0.0       0.0         0       0.01.00000E20",
        "$#     SFS       SFM       SST       MST      SFST      SFMT       FSF       VSF",
        "       0.0       1.0       0.0       0.0       1.0       1.0       1.0       1.0",
        "$#    SOFT    SOFSCL    LCIDAB    MAXPAR     SBOPT     DEPTH     BSORT    FRCFRQ",
        "         1       0.1         0     1.025       2.0         2         0         1",
        "$#  PENMAX    THKOPT    SHLTHK     SNLOG      ISYM     I2D3D    SLDTHK    SLDSTF",
        "       0.0         0         0         0         0         0       0.0       0.0",
        "$#    IGAP    IGNODPRFAC/MPADTSTIF/MPAR2   UNUSED     UNUSED    FLANGL   CID_RCF",
        "         0         1       0.0       0.0                           0.0         0",
        "$",
        "*SET_PART_ADD_TITLE",
        "GLOBAL_EXCLUDE_MASTER_SET",
        "  70000001",
        "  10000749  16000001  68999988  41000001  66000005  64000001  82000066  15000003",
        "  71000065",
        "*END",
    ]

    return body


def create_files(b_path: Path, doe_name: str = "doe.parquet", sim_name: str = "simulations"):
    """Generate key files from DOE

    Args:
        b_path (Path): directory path
        doe_name (str, optional): name of DOE file. Defaults to "doe.parquet".
        sim_name (str, optional): name of simulation directory. Defaults to "simulations".
    """
    # read doe
    doe = pd.read_parquet(b_path / doe_name)

    # create cases
    case_paths: List[Path] = []

    # create directory
    sim_dir = b_path / sim_name
    sim_dir.mkdir()

    # generate cases
    for idx in tqdm.tqdm(doe.index):
        body = create_key_file(
            rid=idx,
            percentile=doe.loc[idx, "PERC"],
            pab_t_vent=doe.loc[idx, "PAB_Vent_T"],
            pab_m_scal=doe.loc[idx, "PAB_M_Scal"],
            sll=doe.loc[idx, "SLL"],
            pulse_scale=doe.loc[idx, "Pulse_X_Scale"],
            pulse_angle_deg=doe.loc[idx, "Pulse_Angle"],
        )
        case_name = body[2]
        run_dir = sim_dir / case_name
        LOG.debug("Create %s", run_dir)
        run_dir.mkdir(exist_ok=True)
        case_paths.append(run_dir / f"{case_name}.key")
        with open(case_paths[-1], "w") as f:
            f.writelines([line + "\n" for line in body])

    LOG.info("Done")


def main():
    """run"""
    # init
    parser = argparse.ArgumentParser(description="Generate DOE by SOBOL sequence")

    # arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Directory to fetch data from",
        required=True,
    )
    parser.add_argument(
        "--doe_name",
        default="doe.parquet",
        help="DOE file name (default: %(default)s)",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--log_lvl",
        default=logging.INFO,
        help="Log level (default: %(default)s)",
        required=False,
        type=int,
    )

    # parse
    args = parser.parse_args()

    # set log level
    custom_log.init_logger(log_lvl=args.log_lvl)

    # run
    LOG.info("Start File Generation")
    create_files(b_path=args.directory, doe_name=args.doe_name)
    LOG.info("Files Generated")


if __name__ == "__main__":
    main()
