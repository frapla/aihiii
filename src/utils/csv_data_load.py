import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np

project_dir = Path(".").absolute().parent
if project_dir not in set(sys.path):
    sys.path.append(str(project_dir))
from src.utils.Csv import Csv

LOG: logging.Logger = logging.getLogger(__name__)


def load_single_channel(in_tuple: Tuple[Path, str]) -> Union[Tuple[str, np.ndarray], None]:
    path, ch_name = in_tuple
    perc = ch_name[-6:-4]
    if path.parent.is_dir() and path.is_file():
        db = Csv(csv_path=path).read()

        if db.columns[0][-6:-4] == perc:
            return path.parent.stem, db[ch_name].to_numpy()
        else:
            return None
    else:
        LOG.error("File do not exist in %s", path)
        return None
