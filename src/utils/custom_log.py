import logging


class Colors:
    """https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    ANSI color codes"""

    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


class ColorFormatter(logging.Formatter):
    # Change this dictionary to suit your coloring needs!
    COLORS = {
        "DEBUG": Colors.LIGHT_GRAY,
        "INFO": Colors.GREEN,
        "WARNING": Colors.PURPLE,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        if color:
            return color + logging.Formatter.format(self, record) + Colors.END
        else:
            return logging.Formatter.format(self, record)


def init_logger(log_lvl: int = logging.NOTSET) -> logging.Logger:
    """Initialize a logger with colorized output

    Args:
        name (Optional[str], optional): name of logger (usually module name). Defaults to None.
        log_lvl (int, optional): propagate log level if NOTSET. Defaults to logging.NOTSET.
        start_msg (str, optional): Info message at init. Defaults to "START".

    Returns:
        logging.Logger: logger
    """
    f_str = "%(asctime)s %(levelname)8s %(processName)12s %(threadName)11s %(message)s"
    if log_lvl == logging.DEBUG:
        f_str += " || %(module)s, %(lineno)d"
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(f_str))

    logging.basicConfig(level=log_lvl, handlers=[console], force=True)
