import json
import logging
from pathlib import Path, PosixPath, WindowsPath

import numpy as np

LOG: logging.Logger = logging.getLogger(__name__)


class GeneralEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, PosixPath) or isinstance(obj, WindowsPath):
            return str(obj).replace("\\\\", "\\")
        else:
            return json.JSONEncoder.default(self, obj)


def _test_obj(obj):
    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        new_obj = []
        for item in obj:
            item_new = _test_obj(item)
            if item_new is None:
                LOG.warning("Item %s of list %s not JSON serializable - SKIP")
            else:
                new_obj.append(item_new)
        if isinstance(obj, np.ndarray):
            new_obj = np.array(new_obj)
        if len(new_obj) != len(obj):
            LOG.warning("n=%s objects dropped from list %s", len(obj) - len(new_obj), obj)

    elif isinstance(obj, dict):
        new_obj = {}
        for key in obj.keys():
            new_key = _test_obj(key)
            new_items = _test_obj(obj[key])
            if new_key is None:
                LOG.warning("Key %s not JSON serializable - SKIP", key)
            else:
                new_obj[new_key] = new_items
        if len(obj.keys()) != len(new_obj.keys()):
            LOG.warning("n=%s objects dropped from dict %s", len(obj.keys()) - len(new_obj.keys()), obj)
    else:
        try:
            _ = json.dumps(obj=obj, cls=GeneralEncoder)
            new_obj = obj
        except TypeError:
            LOG.warning("Object %s not JSON serializable - set to None", obj)
            new_obj = None

    return new_obj


def _path_ending(f_path: Path):
    return f_path if f_path.suffix == ".json" else f_path.parent / f"{f_path.stem}.json"


def dump_with_hash(obj, f_path: Path):
    obj_ = _test_obj(obj)
    json_str = json.dumps(obj=obj_, cls=GeneralEncoder)
    f_path = f_path.parent / f"{f_path.stem}_{hash(json_str)}.json"
    dump(obj=obj_, f_path=f_path)


def dump(obj, f_path: Path) -> Path:
    f_path = _path_ending(f_path=f_path)
    LOG.debug(f"Write {f_path}")

    with open(file=f_path, mode="w", encoding="utf-8") as f:
        json.dump(obj=_test_obj(obj), indent=2, fp=f, cls=GeneralEncoder)

    return f_path


def load(f_path: Path):
    f_path = _path_ending(f_path=f_path)
    LOG.debug(f"Load {f_path}")

    with open(file=f_path, mode="r") as f:
        obj = json.load(fp=f)

    return obj
