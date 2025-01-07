import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from zipfile import ZipFile

from lasso.dyna import Binout

LOG: logging.Logger = logging.getLogger(__name__)


class ReadBinout:
    def __init__(
        self,
        sim_dir: Path,
    ) -> None:
        """Handler for binouts

        Args:
            sim_dir (Path): Directory with simulation results
            binout (str, optional): search string to match result file. Defaults to "binout*".
            log (Union[Logger, None], optional): logger. Defaults to None.
        """
        # init logging
        self.bin_path = "binout*"

        # check directory
        if sim_dir.is_dir() and sim_dir.glob(self.bin_path):
            self.source_dir = sim_dir
        else:
            LOG.critical("Directory %s does not exist or does not contain any %s files - EXIT", sim_dir, self.bin_path)

        # init
        self.binout: Union[Binout, None] = None
        self.work_dir: Union[TemporaryDirectory, None] = None

    def __enter__(self):
        """Open file(s)

        Returns:
            _type_: opened file object
        """
        # create temporary directory
        self.work_dir = TemporaryDirectory(prefix=f"{self.source_dir.stem}_")
        work_dir = Path(self.work_dir.name)
        LOG.info("Process simulation from %s in %s", self.source_dir, work_dir)

        # copy files to tmp directory
        zip_path = list(self.source_dir.glob("binout*.zip"))
        if zip_path and zip_path[0].is_file():
            LOG.debug("Extract %s to %s", zip_path[0], work_dir)
            with ZipFile(zip_path[0], "r") as zip_ref:
                zip_ref.extractall(work_dir)
        else:
            for binout_file in self.source_dir.glob(self.bin_path):
                LOG.debug("Copy %s to %s", binout_file, work_dir)
                shutil.copy(binout_file, work_dir)

        # read binout
        self.binout = Binout(filepath=str(work_dir / self.bin_path))
        LOG.info(f"Opened {self.binout.filelist}")

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Close file(s)

        Args:
            exception_type (_type_): not used, for with method
            exception_value (_type_): not used, for with method
            exception_traceback (_type_): not used, for with method
        """
        LOG.debug(f"Close {self.binout.filelist}")
        self.binout.lsda.close()

        LOG.debug("Cleanup %s", self.work_dir.name)
        self.work_dir.cleanup()
