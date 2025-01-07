import logging

LOG: logging.Logger = logging.getLogger(__name__)


class ParallelException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def handler(error: ParallelException):
    """Handler to propagate parallel occurring errors

    Args:
        error (_type_): error message
    """
    LOG.critical("ERROR: %s from %s\n%s", type(error), error.args[1], error.args[0])
