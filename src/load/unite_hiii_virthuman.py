import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
import src.utils.hash_file as hash_file
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.utils.PathChecker import PathChecker

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()
FACTOR_PARSER: Dict[str, str] = {
    "PABPSCAL": "PAB_M_Scal",
    "PABTVENT": "PAB_Vent_T",
    "ALPHA": "Pulse_Angle",
    "PSCAL": "Pulse_X_Scale",
    "SLL": "SLL",
}


def eval_cmd() -> Tuple[Path, Path]:
    # cmd line
    parser = argparse.ArgumentParser(description="Combine Data")
    parser.add_argument("--dir_hiii", type=Path, help="Directory with data for HIII", required=True)
    parser.add_argument("--dir_vh", type=Path, help="Directory with data for VIRTHUMAN", required=True)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    # logging
    if args.verbose:
        custom_log.init_logger(log_lvl=logging.DEBUG)
    else:
        custom_log.init_logger(log_lvl=logging.INFO)

    # paths
    dir_hiii = PathChecker().check_directory(args.dir_hiii)
    dir_vh = PathChecker().check_directory(args.dir_vh)

    return dir_hiii, dir_vh


def read_doe(doe_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read DOE type file

    Args:
        doe_dir (Path): directory with doe.parquet and optional sim_id_2_id.parquet

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: DOE with 3 levels in unit space (index are factor, value is ID), hash of input files (key is path)
    """
    LOG.info("Reading Data from %s", doe_dir)

    # init path and hash
    doe_fpath = PathChecker().check_file(doe_dir / "doe.parquet")
    in_hashes = {str(doe_fpath): hash_file.hash_file(fpath=doe_fpath)}

    if STR.sim_id in pl.read_parquet_schema(doe_fpath):
        # get DOE with SIM_ID
        doe = pl.scan_parquet(doe_fpath).filter(pl.col(STR.perc).eq(5)).select(pl.exclude(STR.perc))

        # get parser for ID to SIM_ID
        parser_path = PathChecker().check_file(doe_dir / "sim_id_2_id.parquet")
        in_hashes[str(parser_path)] = hash_file.hash_file(fpath=parser_path)
        parser = pl.scan_parquet(parser_path).filter(pl.col(STR.perc).eq(5)).select(pl.exclude(STR.perc))

        # join and drop SIM_ID
        doe = doe.join(parser, on=STR.sim_id, how="inner").select(pl.exclude(STR.sim_id)).collect().to_pandas().set_index(STR.id)

    else:
        # get DOE with ID
        doe = pl.scan_parquet(doe_fpath).rename(FACTOR_PARSER).collect().to_pandas().set_index(STR.id)

    # remove all samples with factor levels that are not min, max or median (3 level full factorial)
    LOG.info("Reduce DOE from %s", doe.shape)
    for col in doe.columns:
        # generate non min, max, median level mask
        uniques = sorted(doe[col].unique())
        droppers = set(uniques) - {min(uniques), max(uniques), np.median(uniques)}
        replacer = {val: np.nan for val in droppers}

        # mask
        doe[col] = doe[col].replace(to_replace=replacer).to_list()

    # drop all rows with NaN
    doe.dropna(inplace=True)
    LOG.info("Reduced DOE to %s", doe.shape)

    # convert to unit space (-1, 0, 1)
    LOG.info("Convert to Unit Space")
    for col in doe.columns:
        uniques = sorted(doe[col].unique())
        replacer = {val: idx for idx, val in enumerate(uniques, -(len(uniques) // 2))}
        doe[col] = doe[col].replace(to_replace=replacer).to_list()
    doe = doe.astype(int)
    LOG.info("Converted to Unit Space: %s", doe.shape)

    # swap columns and index
    LOG.info("Swap Col and index")
    doe.reset_index(inplace=True)
    doe.set_index(sorted(set(doe.columns) - {STR.id}), inplace=True)
    doe.rename(columns={STR.id: doe_dir.stem}, inplace=True)
    LOG.info("Loaded Data from %s:\n%s", doe_fpath, doe)

    return doe, in_hashes


def get_fe_simulation_data(db_path: Path, id_renamer: Dict[int, int], up: int = 0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load data

    Args:
        db_path (Path): path to database
        id_renamer (Dict[int, int]): key is generalized ID, value is ID in database
        up (int, optional): Add to percentile(s). Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: database, hash of input files (key is path)
    """
    # get path and hash
    LOG.info("Hash Data from %s", db_path)
    db_fpath = PathChecker().check_file(db_path)
    in_hashes = {str(db_fpath): hash_file.hash_file(fpath=db_fpath)}

    # droppers
    drop_fpath = db_path.parent / STR.fname_dropped_ids
    if drop_fpath.is_file():
        drops = {int(p): d for p, d in json_util.load(f_path=drop_fpath).items()}
        in_hashes[str(drop_fpath)] = hash_file.hash_file(fpath=drop_fpath)
    else:
        drops = {5: [], 50: [], 95: []}
    LOG.info("Dropped IDs:\n%s", drops)

    LOG.debug("Renamer Keys: %s", sorted(id_renamer.keys()))
    LOG.debug("Renamer Values: %s", sorted(id_renamer.values()))

    if STR.time in pl.read_parquet_schema(db_fpath):
        LOG.info("Load Temporal Data")
        db = (
            pl.scan_parquet(db_fpath)
            .filter(~(pl.col(STR.perc).eq(5) & pl.col(STR.id).is_in(drops[5])))
            .filter(~(pl.col(STR.perc).eq(50) & pl.col(STR.id).is_in(drops[50])))
            .filter(~(pl.col(STR.perc).eq(95) & pl.col(STR.id).is_in(drops[95])))
            .filter(pl.col(STR.id).is_in(id_renamer.keys()))
            .select(
                pl.col(STR.id).replace(id_renamer).cast(pl.Int32),
                pl.col(STR.perc).replace({} if up == 0 else {p: p + up for p in (5, 50, 95)}).cast(pl.Int32),
                pl.col(STR.time).round(3),
                pl.exclude(STR.id, STR.perc, STR.time),
            )
            .collect()
            .to_pandas()
            .set_index([STR.time, STR.id, STR.perc])
            .sort_index()
        )

    else:
        db = (
            pl.scan_parquet(db_fpath)
            .filter(~(pl.col(STR.perc).eq(5) & pl.col(STR.id).is_in(drops[5])))
            .filter(~(pl.col(STR.perc).eq(50) & pl.col(STR.id).is_in(drops[50])))
            .filter(~(pl.col(STR.perc).eq(95) & pl.col(STR.id).is_in(drops[95])))
            .filter(pl.col(STR.id).is_in(id_renamer.keys()))
            .select(
                pl.col(STR.id).replace(id_renamer).cast(pl.Int32),
                pl.col(STR.perc).replace({} if up == 0 else {p: p + up for p in (5, 50, 95)}).cast(pl.Int32),
                pl.exclude(STR.id, STR.perc),
            )
            .collect()
            .to_pandas()
            .set_index([STR.id, STR.perc])
            .sort_index()
        )

    LOG.info("Loaded Data from %s:\n%s", db_fpath, db)

    for idx_name in db.index.names:
        LOG.debug("Unique %s:\n%s", idx_name, sorted(db.index.get_level_values(idx_name).unique()))
    LOG.debug("Duplications:\n%s", db[db.index.duplicated(keep=False)])

    return db, in_hashes


def main(dir_hiii: Path, dir_vh: Path):
    # init hash collector
    in_hashs_hiii, in_hashs_vh = {}, {}
    out_dir = Path("data") / "doe" / "unite_hiii_virthuman"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load DOE data
    LOG.info("Load DOE Data")
    doe_hiii, doe_hiii_in_hashs = read_doe(dir_hiii)
    doe_vh, doe_vh_in_hashs = read_doe(dir_vh)

    # update hashs
    in_hashs_hiii.update(doe_hiii_in_hashs)
    in_hashs_vh.update(doe_vh_in_hashs)

    # Generate parser databases' IDs to Generalized IDs
    LOG.info("Join")
    parser = doe_hiii.join(doe_vh, how="inner")
    del doe_hiii, doe_vh
    parser.index = list(range(parser.shape[0]))
    parser.index.name = STR.id
    LOG.info("Got %s with:\n%s", parser.shape, parser)

    # Load Channel Data HIII
    LOG.info("Load Channel Data HIII")
    chs_hiii, chs_hiii_in_hashs = get_fe_simulation_data(
        db_path=dir_hiii / STR.fname_channels,
        id_renamer={v: k for k, v in parser[dir_hiii.stem].to_dict().items()},
    )
    in_hashs_hiii.update(chs_hiii_in_hashs)
    LOG.info("Loaded Channel Data HIII")

    # Load Channel Data VIRTHUMAN
    LOG.info("Load Channel Data VIRTHUMAN")
    chs_vh, chs_vh_in_hashs = get_fe_simulation_data(
        db_path=dir_vh / STR.fname_channels,
        id_renamer={v: k for k, v in parser[dir_vh.stem].to_dict().items()},
        up=1,
    )
    in_hashs_vh.update(chs_vh_in_hashs)
    LOG.info("Loaded Channel Data VH")

    # concatenate channel data
    LOG.info("Concatenate Channel HIII and VH")
    chs = pd.concat([chs_hiii, chs_vh]).dropna(axis=1).sort_index()
    del chs_hiii, chs_vh
    LOG.info("Got %s with:\n%s", chs.shape, chs)

    # save channel data
    LOG.info("Save Channel Data")
    out_fpath_ch = out_dir / STR.fname_channels
    chs.to_parquet(out_fpath_ch, index=True)
    del chs
    ch_hash = hash_file.hash_file(fpath=out_fpath_ch)
    LOG.info("Saved Data to %s", out_fpath_ch)

    # Load Injury Data HIII
    LOG.info("Load Injury Data HIII")
    inj_hiii, inj_hiii_in_hashs = get_fe_simulation_data(
        db_path=dir_hiii / STR.fname_injury_crit,
        id_renamer={v: k for k, v in parser[dir_hiii.stem].to_dict().items()},
    )
    in_hashs_hiii.update(inj_hiii_in_hashs)
    LOG.info("Loaded Injury Data HIII")

    # Load Injury Data VIRTHUMAN
    LOG.info("Load Injury Data VIRTHUMAN")
    inj_vh, inj_vh_in_hashs = get_fe_simulation_data(
        db_path=dir_vh / STR.fname_injury_crit,
        id_renamer={v: k for k, v in parser[dir_vh.stem].to_dict().items()},
        up=1,
    )
    in_hashs_vh.update(inj_vh_in_hashs)
    LOG.info("Loaded Injury Data VH")

    # concatenate injury data
    LOG.info("Concatenate Injury HIII and VH")
    inj = pd.concat([inj_hiii, inj_vh]).dropna(axis=1).sort_index()
    del inj_hiii, inj_vh
    LOG.info("Got %s with:\n%s", inj.shape, inj)

    # save injury data
    LOG.info("Save Injury Data")
    out_fpath_inj = out_dir / STR.fname_injury_crit
    inj.to_parquet(out_fpath_inj, index=True)
    del inj
    inj_hash = hash_file.hash_file(fpath=out_fpath_inj)
    LOG.info("Saved Data to %s", out_fpath_inj)

    # bookkeeping
    LOG.info("Save Bookkeeping")
    book = {
        STR.creation: datetime.datetime.now().isoformat(),
        "Input": {
            "HIII": in_hashs_hiii,
            "VH": in_hashs_vh,
        },
        "Output": {
            STR.channelss: {
                STR.path: out_fpath_ch,
                STR.hash: ch_hash,
            },
            "Injury Criteria": {
                STR.path: out_fpath_inj,
                STR.hash: inj_hash,
            },
        },
        STR.perc: {"HIII": [5, 95, 50], "VH": [6, 96, 51]},
    }
    json_util.dump(obj=book, f_path=out_dir / STR.fname_results_info)
    LOG.info("Saved Bookkeeping to %s", out_dir / STR.fname_results_info)


if __name__ == "__main__":
    main(*eval_cmd())
