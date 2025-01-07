from typing import Tuple


class SimProperty:
    def __init__(
        self,
        perc: int = 50,
        pos: str = "03",
        dbuild: str = "H3",
        tend_ms: float = 140,
        sr_ms: float = 0.1,
        units: Tuple[str, str, str] = ("t", "mm", "s"),
    ) -> None:
        """Basic properties of simulation

        Args:
            perc (int, optional): dummy percentile. Defaults to 50.
            pos (str, optional): dummy position (e.g. driver 01). Defaults to "03".
            dbuild (str, optional): dummy type. Defaults to "H3".
            tend_ms (float, optional): expected end time of simulation. Defaults to 140.
            sr_ms (float, optional): expected sampling rate of signals. Defaults to 0.1.
            units (Tuple[str, str, str], optional): input unit system. Defaults to ("t", "mm", "s").
        """
        self.dummy_percentile = perc
        self.dummy_position = pos
        self.dummy_type = dbuild
        self.tend_ms = tend_ms
        self.sampling_rate_ms = sr_ms
        self.unit_system_in = units
