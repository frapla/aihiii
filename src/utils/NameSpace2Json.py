import logging
from pathlib import Path

import json_util

LOG: logging.Logger = logging.getLogger(__name__)


class NameSpace2Json:
    def __init__(self, prefix: str = "_") -> None:
        self.prefix = prefix
        self.attributes = {}

    def get_attr(self, class_obj: object):
        attributes_all = class_obj.__dict__
        for attr in attributes_all.keys():
            if len(attr) > 2 and attr[0] == self.prefix and attr[1] != self.prefix:
                self.attributes[attr] = attributes_all[attr]
        LOG.debug("Found %s attributes", len(self.attributes))

    def to_json(self, f_path: Path) -> Path:
        return json_util.dump(obj=self.attributes, f_path=f_path)
