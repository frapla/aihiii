import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
from sklearn.metrics.pairwise import pairwise_distances

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.utils.custom_log import init_logger
from src._StandardNames import StandardNames
from src.utils.PathChecker import PathChecker
from src.build.BaseLineUniversal import BaseLineUniversal
from src.experiments._Experiment import Experiment
from src.utils import json_util

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class KnnSampler:
    def __init__(
        self,
        work_dir: Path,
        data_dir: Path,
        doe_fname: str,
        sobol_m: int,
        n_seeds: int,
        n_neighbors: int,
        strategy: Literal["max", "mean", "median"],
        experiment_kwargs: dict,
        score_threshold: float,
        max_iterations: int = 100000,
        doe_ref_perc: int = 5,
    ):
        # paths
        path_checker = PathChecker()
        self.data_dir: Path = path_checker.check_directory(data_dir)
        self.doe_fpath: Path = path_checker.check_file(self.data_dir / doe_fname)
        self.dropped_id_fpath: Path = path_checker.check_file(self.data_dir / STR.fname_dropped_ids)
        self.sim_id_2_id_fpath: Path = path_checker.check_file(self.data_dir / STR.fname_sim_id_2_id)
        self.sampler_work_dir: Path = path_checker.check_directory(work_dir)
        self.book_fpath: Path = self.sampler_work_dir / STR.fname_results
        self.stop_fpath: Path = self.sampler_work_dir / "stop.txt"

        # parameter
        self.sobol_m: int = sobol_m
        self.n_seeds: int = n_seeds
        self.n_neighbors: int = n_neighbors
        self.strategy: Literal["max", "mean", "median"] = strategy
        self.doe_ref_perc: int = doe_ref_perc
        self.score_threshold: float = score_threshold
        self.max_iterations: int = max_iterations

        # init
        self._distance_matrix: Optional[pd.DataFrame] = None
        self._ids_for_training: Optional[List[int]] = None
        self._experiment_kwargs: dict = experiment_kwargs
        self._iteration: int = 0

        # book
        if self.book_fpath.is_file():
            LOG.info("Resume study from %s", self.book_fpath)
            self._book: dict = json_util.load(self.book_fpath)
            self._iteration = len(self._book[STR.id])
            self._ids_for_training: List[int] = self._book[STR.id][-1]
        else:
            LOG.info("Start new study")
            self._book: dict = {
                STR.creation: datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                STR.info: {key: val for key, val in self.__dict__.items() if not key.startswith("_")},
                STR.id: [],
                STR.metrics: [],
                STR.assess_sample_ids: [],
                STR.assess_error: [],
            }

    def run(self) -> None:
        # get basic data
        self._init_distance_matrix()

        # set initial ids
        self._init_samples()

        # loop
        while (
            (len(self._book[STR.metrics]) == 0 or self._book[STR.metrics][-1] < self.score_threshold)
            and self._iteration < self.max_iterations
            and len(self._ids_for_training) < self._distance_matrix.shape[0]
            and not self.stop_fpath.is_file()
        ):
            LOG.info("Run Iteration %s", self._iteration)
            initial_fill_ids = self._training()

            # book
            LOG.info("Save book for iteration %s", self._iteration)
            self._save_book()

            # next ids
            self._select_new_sim_ids(reinforce_ids=initial_fill_ids)

            LOG.info("Iteration %s done", self._iteration)
        else:
            LOG.info("Stop criteria reached")
            if self.stop_fpath.is_file():
                LOG.info("Stop file found")
                self.stop_fpath.unlink()

    def _save_book(self) -> None:
        # book
        json_util.dump(obj=self._book, f_path=self.book_fpath)

        # progression
        # matplotlib backend for server (suppress TKinter)
        matplotlib.use("Agg")
        fig, ax = plt.subplots()
        ax.plot(
            [len(ids) for ids in self._book[STR.id]],
            [max_error for max_error in self._book[STR.metrics]],
            marker="o",
            label="Progression",
        )
        ax.axhline(y=1, label="Theoretical Best", color="red", linestyle="--")
        ax.axhline(y=self.score_threshold, label="Stop Threshold", color="green", linestyle="--")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Mean Error")
        ax.set_title(f"Progression of {__name__}")
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid()
        ax.legend()
        fig.savefig(self.book_fpath.with_suffix(".png"))
        plt.close(fig)

    def _select_new_sim_ids(self, reinforce_ids: List[int]) -> List[int]:
        LOG.info("Shorten distance matrix %s", self._distance_matrix.shape)
        self._distance_matrix.drop(columns=self._ids_for_training, inplace=True, errors="ignore")
        LOG.info("Shortened distance matrix %s", self._distance_matrix.shape)

        if self._distance_matrix.shape[1] == 0:
            LOG.info("All ids used")
            with open(self.stop_fpath, "w") as f:
                f.write("All ids used")
        else:
            LOG.info("Select new sim_ids from distance matrix")
            new_ids = []
            for reinforce_id in reinforce_ids:
                new_from_reinf = self._distance_matrix.loc[reinforce_id].nsmallest(self.n_neighbors).index.to_list()
                self._distance_matrix.drop(columns=new_from_reinf, inplace=True, errors="ignore")
                new_ids.extend(new_from_reinf)
            LOG.info("Add %s new sim_ids", len(new_ids))
            LOG.debug("New sim_ids: %s from %s", new_ids, reinforce_ids)

            self._ids_for_training.extend(new_ids)
            len_ids = len(self._ids_for_training)
            self._ids_for_training = sorted(set(self._ids_for_training))
            len_ids_unique = len(self._ids_for_training)
            if len_ids != len_ids_unique:
                LOG.error("Removed duplicates - %s -> %s", len_ids, len_ids_unique)
            LOG.info("Use n ids for training: %s", len(self._ids_for_training))

    def _training(self) -> List[int]:
        LOG.info("Training Iteration %s", self._iteration)
        work_dir = self.sampler_work_dir / f"iteration_{self._iteration}"
        work_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()

        LOG.info("Run Experiment in %s", work_dir)
        os.chdir(work_dir)
        self._experiment_kwargs["used_ids_ai"] = self._ids_for_training.copy()
        self._book[STR.id].append(self._ids_for_training.copy())
        experiment = Experiment(**self._experiment_kwargs)
        experiment.prepare()
        experiment.run()
        self._book[STR.metrics].append(experiment.get_score())
        os.chdir(cwd)
        LOG.info("Experiment done - score: %s", self._book[STR.metrics][-1])

        LOG.info("Get score")
        sample_score: pd.DataFrame = pd.read_parquet(work_dir / STR.fname_sample_score)
        if self.strategy == "max":
            sample_score: float = sample_score.max(axis=1)
        elif self.strategy == "mean":
            sample_score: float = sample_score.mean(axis=1)
        elif self.strategy == "median":
            sample_score: float = sample_score.median(axis=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        max_error_ids = sample_score.nlargest(self.n_seeds).index.to_list()
        self._book[STR.assess_sample_ids].append(max_error_ids)
        self._book[STR.assess_error].append(sample_score.loc[max_error_ids].tolist())
        self._iteration += 1

        LOG.info("Got %s Max error ids - current max error is %s", len(max_error_ids), self._book[STR.metrics][-1])
        LOG.debug("Max error ids: %s", max_error_ids)

        return max_error_ids

    def _init_samples(self) -> None:
        LOG.info("Set initial samples")
        n_samples_init = 2**self.sobol_m
        if n_samples_init > self._distance_matrix.shape[0]:
            raise ValueError(
                f"Number of initial samples {n_samples_init} is larger than DOE size {self._distance_matrix.shape[0]}"
            )

        if self._ids_for_training is None:
            self._ids_for_training: List[int] = self._distance_matrix.index[:n_samples_init].tolist()

    def _init_distance_matrix(self) -> None:
        # get parser
        sim_id_2_id = self._get_parser()

        # load DOE
        LOG.info("Loading DOE from %s", self.doe_fpath)
        doe = (
            pd.read_parquet(self.doe_fpath, filters=[(STR.perc, "==", self.doe_ref_perc)])
            .apply(pd.to_numeric, downcast="float")
            .drop(columns=[STR.perc])
        )

        doe.index = [int(sim_id_2_id[sim_id]) for sim_id in doe.index]
        doe.index.name = STR.id
        LOG.info("DOE shape: %s", doe.shape)
        LOG.debug("DOE:\n%s", doe)

        # clean DOE
        LOG.info("Get drop ids")
        drop_ids: Dict[int, int] = json_util.load(self.dropped_id_fpath)
        for perc in self._experiment_kwargs["feature_percentiles"] + self._experiment_kwargs["target_percentiles"]:
            doe.drop(index=drop_ids[str(perc)], inplace=True)
        LOG.info("DOE shape after dropping: %s", doe.shape)

        # calculate distance matrix
        dst_matrix = pd.DataFrame(
            pairwise_distances(doe.to_numpy(), metric="euclidean", n_jobs=-1),
            columns=doe.index,
            index=doe.index,
        )
        LOG.info("Distance matrix shape: %s", dst_matrix.shape)
        LOG.debug("Distance matrix:\n%s", dst_matrix)

        # store
        self._distance_matrix: pd.DataFrame = dst_matrix

    def _get_parser(self) -> Dict[int, int]:
        LOG.info("Loading parser from %s", self.sim_id_2_id_fpath)
        parser = (
            pd.read_parquet(self.sim_id_2_id_fpath, filters=[(STR.perc, "==", self.doe_ref_perc)])
            .apply(pd.to_numeric, downcast="float")
            .drop(columns=[STR.perc])
        )
        LOG.info("Parser shape: %s", parser.shape)
        LOG.debug("Parser:\n%s", parser)

        LOG.info("Convert parser to dict")
        id_2_sim_id = parser.reset_index().set_index(STR.id)

        return {int(id_2_sim_id.loc[idx, STR.sim_id]): int(idx) for idx in id_2_sim_id.index}


if __name__ == "__main__":
    init_logger(log_lvl=logging.DEBUG)
    LOG.info("Start KnnSampler")
    sampler = KnnSampler(
        data_dir=Path("data") / "doe" / "doe_sobol_20240705_194200",
        exp_dir=Path("experiments"),
        doe_fname="doe_combined.parquet",
        sobol_m=10,
        n_seeds=4,
        n_neighbors=4,
        strategy="max",
        experiment_kwargs={
            "user_pipeline": BaseLineUniversal,
            "processed_data_dir": Path("..") / ".." / ".." / "data" / "doe" / "doe_sobol_20240705_194200",
            "file_names_ai_in": ["injury_criteria"],
            "file_names_ai_out": ["injury_criteria"],
            "feature_percentiles": [50],
            "target_percentiles": [95],
            "used_columns_ai_out": [
                "03HEADLOC0OCCUDSXD",
                "03HEADLOC0OCCUDSYD",
                "03HEADLOC0OCCUDSZD",
                "03HEAD0000OCCUACXD",
                "03HEAD0000OCCUACYD",
                "03HEAD0000OCCUACZD",
                "03CHSTLOC0OCCUDSXD",
                "03CHSTLOC0OCCUDSYD",
                "03CHSTLOC0OCCUDSZD",
                "03CHST0000OCCUDSXD",
                "03CHST0000OCCUACXD",
                "03CHST0000OCCUACYD",
                "03CHST0000OCCUACZD",
                "03PELVLOC0OCCUDSXD",
                "03PELVLOC0OCCUDSYD",
                "03PELVLOC0OCCUDSZD",
                "03PELV0000OCCUACXD",
                "03PELV0000OCCUACYD",
                "03PELV0000OCCUACZD",
                "03NECKUP00OCCUFOXD",
                "03NECKUP00OCCUFOZD",
                "03NECKUP00OCCUMOYD",
                "03FEMRRI00OCCUFOZD",
                "03FEMRLE00OCCUFOZD",
                "Head_HIC15",
                "Head_HIC36",
                "Head_a3ms",
                "Neck_Nij",
                "Neck_Fz_Max_Compression",
                "Neck_Fz_Max_Tension",
                "Neck_My_Max",
                "Neck_Fx_Shear_Max",
                "Chest_Deflection",
                "Chest_a3ms",
                "Femur_Fz_Max_Compression",
                "Femur_Fz_Max_Tension",
                "Femur_Fz_Max",
            ],
            "hyperparameter": {},
            "shuffle_data": True,
            "random_state_shuffle": 42,
        },
        score_threshold=1,
        max_iterations=3,
    )
    sampler.run()
