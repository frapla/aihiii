import datetime
import logging
import subprocess
import sys
from pathlib import Path
import polars as pl
from typing import List, Optional

SRC_DIR = str(Path(__file__).absolute().parents[2])
if SRC_DIR not in set(sys.path):
    sys.path.append(SRC_DIR)
import src.utils.json_util as json_util
from src._StandardNames import StandardNames

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


def column_checker(data_dir: Path, file_names: List[str], used_columns: Optional[List[str]] = None) -> List[str]:
    """Generate list of columns to use for ai

    Args:
        data_dir (Path): directory of standardized data files
        file_names (List[str]): names of files to check for columns
        used_columns (Optional[List[str]], optional): columns to use (all available if None). Defaults to None.

    Returns:
        List[str]: sorted list of columns to use
    """
    available_cols = set([])
    for file_name in file_names:
        available_cols |= set(pl.read_parquet_schema((data_dir / file_name).with_suffix(".parquet")).keys())
    available_cols -= {STR.id, STR.time, STR.perc}

    if used_columns is None:
        used_columns = sorted(available_cols)
    else:
        used_columns = sorted(set(used_columns) & available_cols)

    return used_columns


class Parameters:
    def __init__(self) -> None:
        pass

    def create(
        self,
        exp_dir: Path,
        data_dir: Path,
        pipeline_paras: dict,
        file_names_ai_in: List[str],
        file_names_ai_out: List[str],
        feature_percentiles: Optional[List[int]] = None,
        target_percentiles: Optional[List[int]] = None,
        used_columns_ai_in: Optional[List[str]] = None,
        used_columns_ai_out: Optional[List[str]] = None,
        used_ids_ai: Optional[List[int]] = None,
    ):
        """Create parameter.json with standardized entries and  pipeline specifics

        Args:
            exp_dir (Path): directory of experiment, used for output
            data_dir (Path): directory of standardized data files
            pipeline_paras (dict): pipeline specific parameters compatible with .set_params method
            file_names_ai_in (List[str]): file names of data to feed into ai as input
            file_names_ai_out (List[str]): file names of data to be predicted by ai
            feature_percentiles (Optional[List[int]], optional): dummy percentile(s) used for feature, None if features are DOE factors. Defaults to None.
            target_percentiles (Optional[List[int]], optional): dummy percentile(s) used for target, None if target are DOE factors. Defaults to None.
            used_columns_ai_in (Optional[List[str]], optional): columns to use as feature, use all if None. Defaults to None.
            used_columns_ai_out (Optional[List[str]], optional): columns to use as target, use all if None. Defaults to None.
            used_ids_ai (Optional[List[int]], optional): selection of simulation ids to use, use all if None. Defaults to None.
        """
        LOG.info("Store experiment's parameter")
        parameters = {
            STR.creation: str(datetime.datetime.now()),
            STR.data: {
                STR.input: {
                    STR.dir: data_dir,
                    STR.feature: file_names_ai_in,
                    STR.target: file_names_ai_out,
                },
                STR.output: exp_dir,
            },
            STR.perc: {
                STR.feature: feature_percentiles,
                STR.target: target_percentiles,
            },
            STR.channels: {
                STR.feature: column_checker(data_dir, file_names_ai_in, used_columns_ai_in),
                STR.target: column_checker(data_dir, file_names_ai_out, used_columns_ai_out),
            },
            STR.id: used_ids_ai,
            STR.pipeline: pipeline_paras,
            STR.python: {
                "Python": sys.version,
                "Git_Parent": {
                    "Hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SRC_DIR).decode("ascii").strip(),
                    "Branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=SRC_DIR)
                    .decode("ascii")
                    .strip(),
                },
            },
        }

        json_util.dump(obj=parameters, f_path=exp_dir / STR.fname_para)

    def read(self, exp_dir: Path) -> dict:
        """Read and check parameters.json

        Args:
            exp_dir (Path): directory with parameters.json

        Returns:
            dict: parameters
        """
        f_path = exp_dir / STR.fname_para
        paras = json_util.load(f_path=f_path)

        return paras
