import datetime
import logging
import sys
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Type, Dict
from itertools import product

import pandas as pd
import polars as pl
from IPython.display import display

src_dir = str(Path(__file__).absolute().parents[1])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.json_util as json_util
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.evaluate._Evaluate import Evaluate
from src.experiments._Parameters import Parameters
from src.utils._ObjectChecker import ObjectChecker
from src.utils.hash_file import hash_file

LOG: logging.Logger = logging.getLogger(__name__)
STR: StandardNames = StandardNames()


class Pipeline:
    def __init__(
        self,
        model_pipe: Type[BasePipe],
        random_state_shuffle: Optional[int] = 42,
        shuffle_data: bool = True,
    ) -> None:
        """Wrapper for user pipeline to run training and evaluation

        Args:
            model_pipe (Type[BasePipe]): initialized user pipeline
            random_state (Union[None, int], optional): random state of KFOLD shuffle. Defaults to 42.
            shuffle_data (bool, optional): controls KFOLD shuffle. Defaults to True
        """
        # set environment
        self.parameter_fpath: Path = self.__check_file(fpath=Path(STR.fname_para))
        self.parameter_hash: str = hash_file(fpath=self.parameter_fpath)
        self.results_fpath: Path = self.__check_results_file(fpath=Path(STR.fname_results))
        self.results_csv_fpath: Path = self.__check_results_file(fpath=Path(STR.fname_results_csv))
        self.in_data_dir: Path = Path()
        self.file_paths_ai_in: List[Path] = []
        self.file_paths_ai_out: List[Path] = []
        self.target_percentiles: Optional[List[int]] = None
        self.feature_percentiles: Optional[List[int]] = None
        self.used_columns_ai_in: Optional[List[str]] = None
        self.used_columns_ai_out: Optional[List[str]] = None
        self.case_ids_to_select: Optional[List[int]] = None

        # get pipe
        self.pipe: Type[BasePipe] = ObjectChecker().pipeline(pipe=model_pipe)
        self.pipe_paras: dict = {}
        self.random_state_shuffle: Optional[int] = random_state_shuffle
        self.shuffle_data: bool = shuffle_data

        # init data
        self.case_ids: List[int] = []
        self.evaluator: Optional[Evaluate] = None

    def run(self) -> None:
        """Run Pipeline"""
        LOG.info("Start Experiment")
        LOG.info("Load Parameters")
        self.__load_parameters()
        LOG.info("Load Data")
        self.__load_data()
        LOG.info("Evaluate")
        self.__evaluate()
        LOG.info("Store")
        self.__store()
        LOG.info("Experiment End")

    def __check_directory(self, dir_path: Path) -> Path:
        """Check if directory exists

        Args:
            dir_path (Path): path to be checked

        Returns:
            Path: checked path
        """
        if dir_path.is_dir():
            LOG.debug("Directory is %s", dir_path)
        else:
            LOG.critical("Directory %s does not exist - EXIT", dir_path)
            sys.exit()

        return dir_path

    def __check_file(self, fpath: Path) -> Path:
        """Check is file exists

        Args:
            fpath (Path): path of file to be checked

        Returns:
            Path: checked path
        """
        if fpath.is_file():
            LOG.debug("File is %s", fpath)
        else:
            LOG.critical("File %s does not exist - EXIT", fpath)
            sys.exit()

        return fpath

    def __check_results_file(self, fpath: Path) -> Path:
        """Check if result file exist - delete if True

        Args:
            fpath (Path): path to result file

        Returns:
            Path: path of result clean file
        """
        if fpath.is_file():
            LOG.warning("Results file %s already exist - REMOVE", fpath)
            fpath.unlink()
        else:
            LOG.debug("Results file in %s", fpath)

        return fpath

    def __load_parameters(self) -> None:
        """Load parameters from json file to model pipeline"""
        LOG.info("Load parameters to model pipeline")
        # read json
        para = Parameters()
        paras = para.read(exp_dir=Path())

        # data directory
        self.in_data_dir = self.__check_directory(Path(paras[STR.data][STR.input][STR.dir]))

        # data version
        info_file = self.__check_file(fpath=self.in_data_dir / "results_info.json")
        file_hashes = {
            PureWindowsPath(val[STR.path]).name: val[STR.hash] for val in json_util.load(f_path=info_file)[STR.output].values()
        }

        # store paths
        for f_name in paras[STR.data][STR.input][STR.feature]:
            f_path = self.__check_file((self.in_data_dir / f_name).with_suffix(STR.parquet))
            f_hash = hash_file(fpath=f_path)
            if f_hash == file_hashes[f_path.name]:
                LOG.info("IN: Hash %s for %s matches with hash from %s ", f_hash, f_path.name, info_file.name)
                self.file_paths_ai_in.append(f_path)
            else:
                LOG.error("IN: Hash %s for %s does not match with hash from %s", f_hash, f_path.name, info_file.name)

        for f_name in paras[STR.data][STR.input][STR.target]:
            f_path = self.__check_file((self.in_data_dir / f_name).with_suffix(STR.parquet))
            f_hash = hash_file(fpath=f_path)
            if f_hash == file_hashes[f_path.name]:
                LOG.info("OUT: Hash %s for %s matches with hash from %s ", f_hash, f_path.name, info_file.name)
                self.file_paths_ai_out.append(f_path)
            else:
                LOG.error("OUT: Hash %s for %s does not match with hash from %s", f_hash, f_path.name, info_file.name)

        # store percentiles
        self.target_percentiles = paras[STR.perc][STR.target]
        self.feature_percentiles = paras[STR.perc][STR.feature]
        LOG.info("Target percentiles %s", self.target_percentiles)
        LOG.info("Feature percentiles %s", self.feature_percentiles)

        # store columns
        self.used_columns_ai_in = paras[STR.channels][STR.feature]
        self.used_columns_ai_out = paras[STR.channels][STR.target]
        LOG.info("Used columns for input: %s", "All" if self.used_columns_ai_in is None else len(self.used_columns_ai_in))
        LOG.info("Used columns for prediction: %s", "All" if self.used_columns_ai_out is None else len(self.used_columns_ai_out))

        # store ids
        self.case_ids_to_select = paras[STR.id]
        LOG.info("Selected %s ids", "all" if self.case_ids_to_select is None else len(self.case_ids_to_select))

        # store parameters for model pipeline
        self.pipe_paras = paras[STR.pipeline]

    def __load_data(self) -> None:
        """Load data into the pipeline"""
        # check data consistency
        ios = [[self.file_paths_ai_in, self.feature_percentiles], [self.file_paths_ai_out, self.target_percentiles]]
        num_ids = 0
        shared_ids = None
        i = 0
        for io in ios:
            for f_path, perc in product(*io):
                i += 1
                avail_ids = set(
                    pl.scan_parquet(f_path)
                    .filter(pl.col(STR.perc) == perc)
                    .select(STR.id)
                    .cast(pl.Int32)
                    .unique()
                    .collect()[STR.id]
                )
                if shared_ids is None:
                    shared_ids = avail_ids
                else:
                    shared_ids &= avail_ids
                num_ids += len(avail_ids)
                LOG.debug("File %s has %s ids for perc %s", f_path, len(avail_ids), perc)
        LOG.info("All files have %s ids\n%s", num_ids, shared_ids)

        if len(shared_ids) == num_ids / i:
            LOG.info("All files have same ids")
        else:
            LOG.warning("Files do not have same ids - use only %s shared IDs", len(shared_ids))

        # select ids
        if self.case_ids_to_select is None:
            self.case_ids_to_select = list(shared_ids)
        else:
            shared_ids &= set(self.case_ids_to_select)
            if len(shared_ids) < len(self.case_ids_to_select):
                LOG.warning(
                    "Selected %s ids are not in data (%s)",
                    len(self.case_ids_to_select) - len(shared_ids),
                    set(self.case_ids_to_select) - shared_ids,
                )
            else:
                LOG.info("Selected %s ids", len(shared_ids))

        # remove to drops
        drop_id_path = self.in_data_dir / STR.fname_dropped_ids
        if drop_id_path.is_file():
            did: Dict[int, List[int]] = json_util.load(f_path=drop_id_path)
            dropped_ids = set([])
            for key in did.keys():
                if key in self.target_percentiles or key in self.feature_percentiles:
                    dropped_ids |= set(did[key])

            LOG.warning("Drop %s ids from %s", len(dropped_ids), drop_id_path)
            self.case_ids_to_select = list(shared_ids - dropped_ids)
        else:
            LOG.info("No ids to drop")

        # store
        self.case_ids = sorted(self.case_ids_to_select)

    def __evaluate(self) -> None:
        """Train and Evaluate pipeline"""
        LOG.debug("Init Evaluation")
        self.evaluator = Evaluate(
            file_paths_ai_in=self.file_paths_ai_in,
            file_paths_ai_out=self.file_paths_ai_out,
            feature_percentiles=self.feature_percentiles,
            target_percentiles=self.target_percentiles,
            used_columns_ai_in=self.used_columns_ai_in,
            used_columns_ai_out=self.used_columns_ai_out,
            case_ids=self.case_ids,
            pipe=self.pipe,
            pipe_paras=self.pipe_paras,
            shuffle=self.shuffle_data,
            random_state=self.random_state_shuffle,
        )

        LOG.debug("Run Evaluation")
        self.evaluator.run()

    def __store(self) -> None:
        """Store results"""
        LOG.info("Store results")
        # metrics results
        self.evaluator.results.get_scores().to_csv(self.results_csv_fpath, index=True)

        # store results
        results = {
            STR.creation: str(datetime.datetime.now()),
            STR.feature: self.feature_percentiles,
            STR.target: self.target_percentiles,
            STR.data: {
                STR.input: {
                    STR.feature: {
                        STR.files: self.file_paths_ai_in,
                        STR.hash: [hash_file(fpath=f) for f in self.file_paths_ai_in],
                    },
                    STR.target: {
                        STR.files: self.file_paths_ai_out,
                        STR.hash: [hash_file(fpath=f) for f in self.file_paths_ai_out],
                    },
                    STR.para: {
                        STR.path: self.parameter_fpath,
                        STR.hash: self.parameter_hash,
                        "shuffle": self.shuffle_data,
                        "random_state": self.random_state_shuffle,
                    },
                    STR.id: self.case_ids,
                },
                STR.result: {
                    STR.path: self.results_csv_fpath,
                    STR.hash: hash_file(fpath=self.results_csv_fpath),
                },
            },
            STR.result: {
                STR.fold: self.evaluator.results.data_split,
                STR.model: self.evaluator.results.data_info,
                "MCDM": self.evaluator.results.mcdm_criteria.get_assessed_criteria(),
            },
        }

        json_util.dump(obj=results, f_path=self.results_fpath)
