import logging
from pathlib import Path
from typing import List, Literal, Union

LOG: logging.Logger = logging.getLogger(__name__)


class PathObjectNotFound(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PathChecker:
    def __init__(self) -> None:
        """Check path objects for their existence in file system"""

    def check_directory(self, path: Path, exit: bool = True) -> Union[Path, None]:
        """Check if directory exists

        Args:
            path (Path): path to directory
            exit (bool, optional): exit script if True and directory not found. Defaults to True.

        Returns:
            Path: path of directory
        """
        return self.__check_path(path=path, path_type="Directory", exit=exit)

    def check_file(self, path: Path, exit: bool = True) -> Union[Path, None]:
        """Check if file exists

        Args:
            path (Path): file path
            exit (bool, optional): exit script if True and file not found. Defaults to True.

        Returns:
            Path: file path
        """
        return self.__check_path(path=path, path_type="File", exit=exit)

    def check_file_type(self, path: Path, file_pattern: str, exit: bool = True) -> Union[List[Path], None]:
        """Check if directory contains any files matching given pattern

        Args:
            path (Path): path of directory which should contain files
            file_pattern (str): file pattern (pathlib.Path.glob compatible)
            exit (bool, optional): exit script if True and no matching files. Defaults to True.

        Returns:
            List[Path]: paths of existing files matching the pattern
        """
        # check root path
        root: Path = self.check_directory(path=path, exit=exit)

        # look for files
        globbed: List[Path] = sorted(root.glob(file_pattern))

        # measure
        if globbed:
            LOG.info("Found files: %s", globbed)
            return globbed
        else:
            if exit:
                LOG.critical("NoFilesFoundError in '%s' with pattern '%s'- EXIT", root, file_pattern)
                raise PathObjectNotFound()
            else:
                LOG.info("NoFilesFoundError in '%s' with pattern '%s'", root, file_pattern)
                return None

    def __check_path(self, path: Path, path_type: Literal["File", "Directory"], exit: bool) -> Union[Path, None]:
        """Generic path checker

        Args:
            path (Path): path
            path_type (Literal[&quot;File&quot;, &quot;Directory&quot;]): specify path object type
            exit (bool): exit script if True and path object not found

        Returns:
            Path: path
        """
        # check
        exists = path.is_dir() if path_type == "Directory" else path.is_file()

        # measure
        if exists:
            LOG.info("%s '%s' exists", path_type, path)
            return path
        else:
            if exit:
                LOG.critical("%sNotFoundError: %s - EXIT", path_type, path)
                raise PathObjectNotFound()
            else:
                LOG.info("%sNotFoundError: %s", path_type, path)
                return None
