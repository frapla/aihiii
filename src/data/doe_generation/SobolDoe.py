import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.append(str(Path(__file__).absolute().parents[3]))
import src.utils.custom_log as custom_log
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.utils.hash_file import hash_file

LOG: logging.Logger = logging.getLogger(__name__)


class SobolDoe:
    def __init__(self, doe_dir: Path, doe_info_name: str = "factors_info.json", doe_name: str = "doe.parquet") -> None:
        """Generate deterministic SOBOL sampling

        Args:
            doe_dir (Path): directory path
            doe_info_name (str, optional): name of parameter file. Defaults to "factors_info.json".
            doe_name (str, optional): name of DOE file. Defaults to "doe.parquet".
        """
        # init
        self.n_samples_total: Optional[int] = None
        self.factors: Optional[Dict[str, List[float, float]]] = None
        self.previous: Optional[Path] = None
        self.percentiles: Optional[List[int]] = None
        self.str = StandardNames()

        # target directory
        self.doe_dir: Path = doe_dir
        if not self.doe_dir.is_dir():
            LOG.critical("%s is not a directory - EXIT", self.doe_dir)
            sys.exit(1)
        self.doe_path: Path = self.doe_dir / doe_name

        # get info
        self.factors_file: Path = self.doe_dir / doe_info_name
        if self.factors_file.is_file():
            c = json_util.load(f_path=self.factors_file)
            self.__dict__.update(c)
            info = [
                f"n_samples_total: {self.n_samples_total}",
                f"factors: {self.factors}",
                f"percentiles: {self.percentiles}",
                f"previous: {self.previous}",
            ]
            LOG.info("Loaded factors:\n%s", "\n".join(info))
        else:
            LOG.critical("%s is not a file - EXIT", self.factors_file)
            sys.exit(1)

        # sobol parameters
        n_samples_total_sobol = self.n_samples_total // len(self.percentiles)
        self.m: int = int(np.log2(n_samples_total_sobol))
        if self.m != np.log2(n_samples_total_sobol):
            LOG.error("n total samples %s must be a power of 2 - EXIT", n_samples_total_sobol)
            sys.exit(1)
        else:
            LOG.info("n=%s, m=%s", n_samples_total_sobol, self.m)

        # existing DOE file
        if self.previous is not None:
            previous = pd.read_parquet(self.previous)
            self.n_samples_existing = previous.index[-1] + 1
            n_samples_existing_sobol = self.n_samples_existing // len(self.percentiles)
            if int(np.log2(n_samples_existing_sobol)) != np.log2(n_samples_existing_sobol):
                LOG.error("n existing samples %s must be a power of 2 - EXIT", n_samples_existing_sobol)
                sys.exit(1)
            if n_samples_existing_sobol > n_samples_total_sobol:
                LOG.error(
                    "n existing samples %s must be less than n total samples %s - EXIT",
                    self.n_samples_existing,
                    self.n_samples_total,
                )
                sys.exit(1)
        else:
            self.n_samples_existing = 0

    def run(self) -> None:
        """run DOE generation"""
        # generate samples
        samples = self._generate_sample()

        # generate DOE
        df = self._generate_doe(samples=samples)

        # store DOE
        self._store_doe(df=df)

    def _generate_sample(self) -> np.ndarray:
        """Generate sample in unit space

        Returns:
            np.ndarray: sample in unit space, (n_samples, n_factors)
        """
        generator = qmc.Sobol(
            d=len(self.factors.keys()),
            scramble=False,
            seed=None,
            optimization=None,
        )
        samples = generator.random_base2(m=self.m)[self.n_samples_existing // len(self.percentiles) :]
        LOG.info("Generated %s samples", samples.shape)

        return samples

    def _generate_doe(self, samples: np.ndarray) -> pd.DataFrame:
        """Map sample in unit space to factor space

        Args:
            samples (np.ndarray): sample in unit space, (n_samples, n_factors)

        Returns:
            pd.DataFrame: sample in factor space (n_samples * n_percentiles, n_factors)
        """
        # sort
        factors = sorted(self.factors.keys())

        # scale
        samples_scaled = qmc.scale(
            sample=samples,
            l_bounds=[self.factors[factor][0] for factor in factors],
            u_bounds=[self.factors[factor][1] for factor in factors],
        )

        # create DataFrame
        df = pd.DataFrame(
            samples_scaled,
            columns=factors,
        )

        # duplicate doe for percentiles
        df_percs = []
        for p in self.percentiles:
            df_perc = df.copy()
            df_perc["PERC"] = p
            df_percs.append(df_perc)
        df = pd.concat(df_percs)

        # reindex
        df.index = range(self.n_samples_existing, self.n_samples_existing + df.shape[0])
        df.index.name = self.str.id
        return df

    def _store_doe(self, df: pd.DataFrame) -> None:
        """Store DOE

        Args:
            df (pd.DataFrame): sample in factor space (n_samples * n_percentiles, n_factors)
        """
        # store
        df.to_parquet(self.doe_path, index=True)
        LOG.info("Stored DOE in %s", self.doe_path)

        # book
        book = {
            self.str.creation: str(datetime.datetime.now()),
            self.str.input: {
                self.str.path: self.factors_file,
                self.str.hash: hash_file(self.factors_file),
            },
            self.str.output: {
                self.str.path: self.doe_path,
                self.str.hash: hash_file(self.doe_path),
            },
        }
        book_path = self.doe_dir / (self.doe_path.stem + "_info.json")
        LOG.info("Stored book in %s", book_path)
        json_util.dump(f_path=book_path, obj=book)


def main():
    """run"""
    # init
    parser = argparse.ArgumentParser(description="Generate DOE by SOBOL sequence")

    # arguments
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        help="Directory to fetch data from",
        required=True,
    )
    parser.add_argument(
        "--doe_info_name",
        default="factors_info.json",
        help="File with construction information (default: %(default)s)",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--doe_name",
        default="doe.parquet",
        help="DOE file name (default: %(default)s)",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--log_lvl",
        default=logging.INFO,
        help="Log level (default: %(default)s)",
        required=False,
        type=int,
    )

    # parse
    args = parser.parse_args()

    # set log level
    custom_log.init_logger(log_lvl=args.log_lvl)

    # run
    LOG.info("Start Sobol DOE")
    doe = SobolDoe(doe_dir=args.directory, doe_info_name=args.doe_info_name, doe_name=args.doe_name)
    doe.run()
    LOG.info("End Sobol DOE")


if __name__ == "__main__":
    main()
