import logging
import multiprocessing
import pickle
import sys
import time
from itertools import starmap
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src._StandardNames import StandardNames
from src.build._BasePipe import BasePipe
from src.evaluate._Data import Data
from src.evaluate._Metrics import Metrics
from src.evaluate.Results import Results
from src.evaluate.ResultsGlobal import ResultsGlobal
from src.utils._ObjectChecker import ObjectChecker
from src.build._InterfaceNames import InterfaceNames

LOG: logging.Logger = logging.getLogger(__name__)


def is_classification(y: pd.DataFrame) -> bool:
    return set(y.values.flatten()) == set([int(x) for x in y.values.flatten()])


def score(
    y_true: Data,
    y_pred: Data,
    fold_id: int,
    mode: Literal["Train", "Test"],
    n_classes: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    LOG.info("Calculate score for fold %s and mode %s", fold_id, mode)
    metrics = Metrics(fold_id=fold_id, mode=mode)

    # score
    if y_true.get_temporal() is None and isinstance(y_true.get_tabular(), pd.DataFrame):
        if is_classification(y_true.get_tabular()) and is_classification(y_pred.get_tabular()):
            LOG.debug("Classification detected")
            metric = metrics.f1_score(y_true=y_true.get_tabular(), y_pred=y_pred.get_tabular(), n_classes=n_classes)
        else:
            LOG.debug("Regression detected")
            metric = metrics.r2_score(y_true=y_true.get_tabular(), y_pred=y_pred.get_tabular())
    elif isinstance(y_true.get_temporal(), pd.DataFrame) and y_true.get_tabular() is None:
        LOG.debug("Multichannel regression detected")
        metric = metrics.iso1871_score(y_true=y_true.get_temporal(), y_pred=y_pred.get_temporal())
    else:
        LOG.error("Mix of temporal and tabular data not supported - no score is calculated")
        metric = None

    if LOG.getEffectiveLevel() == logging.DEBUG:
        f_path_true = Path(f"y_true_{mode}_Fold{fold_id}.parquet")
        f_path_pred = Path(f"y_pred_{mode}_Fold{fold_id}.parquet")
        LOG.debug("Store prediction in %s and %s", f_path_true, f_path_pred)
        if isinstance(y_true.get_tabular(), pd.DataFrame):
            y_true.get_tabular().to_parquet(f_path_true, index=True)
            y_pred.get_tabular().to_parquet(f_path_pred, index=True)
        else:
            y_true.get_temporal().to_parquet(f_path_true, index=True)
            y_pred.get_temporal().to_parquet(f_path_pred, index=True)

    LOG.info("Got score for fold %s and mode %s:\n%s", fold_id, mode, metric)

    return metric


def data_name(side: Literal["true", "pred"], mode: Literal["Train", "Test"], f_idend: int) -> str:
    return f"y_{side}_{mode}_Fold{f_idend}.pkl"


def store_data(f_idend: int, data: Data, side: Literal["true", "pred"], mode: Literal["Train", "Test"]) -> Path:
    f_path = Path(data_name(side=side, mode=mode, f_idend=f_idend))
    LOG.info("Store Test Prediction in %s", f_path)
    with open(f_path, "wb") as f:
        pickle.dump(data, f)

    return f_path


def read_data(f_path: Path) -> Data:
    LOG.info("Load Data %s", f_path)
    with open(f_path, "rb") as f:
        data: Data = pickle.load(f)

    return data


class TrainTestParaContainer:
    def __init__(
        self,
        pipeline: Type[BasePipe],
        data_identifier: int,
        idxs_train: np.ndarray,
        idxs_test: Optional[np.ndarray],
        file_paths_ai_in: List[Path],
        file_paths_ai_out: List[Path],
        feature_percentiles: Optional[List[int]] = None,
        target_percentiles: Optional[List[int]] = None,
        used_columns_ai_in: Optional[List[str]] = None,
        used_columns_ai_out: Optional[List[str]] = None,
        pipe_paras: Optional[dict] = None,
    ):
        self.pipeline: Type[BasePipe] = pipeline
        self.data_identifier: int = data_identifier
        self.idxs_train: np.ndarray = idxs_train
        self.idxs_test: Optional[np.ndarray] = idxs_test
        self.file_paths_ai_in: List[Path] = file_paths_ai_in
        self.file_paths_ai_out: List[Path] = file_paths_ai_out
        self.feature_percentiles: Optional[List[int]] = feature_percentiles
        self.target_percentiles: Optional[List[int]] = target_percentiles
        self.used_columns_ai_in: Optional[List[str]] = used_columns_ai_in
        self.used_columns_ai_out: Optional[List[str]] = used_columns_ai_out
        self.pipe_paras: Optional[dict] = pipe_paras


def train_test_parallel(para: TrainTestParaContainer) -> str:
    """Run the train and test for a single fold, ready for subprocess

    Args:
        para (TrainTestParaContainer): input parameters
    """
    interface_true = InterfaceNames()
    LOG.info("Start Train Test Fold %s", para.data_identifier)
    # init
    results = Results(fold_id=para.data_identifier)
    pipe = para.pipeline()

    # set hyperparameters
    if para.pipe_paras is not None:
        LOG.info("Set Hyperparameters")
        pipe.set_params(**para.pipe_paras)

    # double data read because Data container is mutable -> avoid double RAM usage and side effects
    for mode in ["Train", "Predict"]:
        LOG.info("Run Training mode %s", mode)
        # Get Training Data
        LOG.info("Get Training Data X")
        x = Data()
        x.set_from_files(
            file_paths=para.file_paths_ai_in,
            percentiles=para.feature_percentiles,
            idxs=para.idxs_train,
            columns=para.used_columns_ai_in,
        )
        LOG.info("Got Training Data X")
        x.log_status(ignore_log_lvl=True)

        LOG.info("Get Training Data Y")
        y_true = Data()
        y_true.set_from_files(
            file_paths=para.file_paths_ai_out,
            percentiles=para.target_percentiles,
            idxs=para.idxs_train,
            columns=para.used_columns_ai_out,
        )
        LOG.info("Got Training Data Y")
        y_true.log_status(ignore_log_lvl=True)
        results.y_true_train_fpath = store_data(f_idend=para.data_identifier, data=y_true, side="true", mode="Train")

        # Fit Pipeline
        if mode == "Train":
            LOG.info("Fit Pipeline")
            interface_true.set_target(y_true)
            interface_true.set_features(x)
            tic_fit = time.perf_counter()
            pipe.fit(x=x, y=y_true)
            results.comp_time_fit = time.perf_counter() - tic_fit

            # store
            if para.data_identifier == -1:
                pipe.store()

        if mode == "Predict":
            # Predict on seen data
            LOG.info("Predict on seen data")
            tic_pred = time.perf_counter()
            y_pred = pipe.predict(x=x)
            if not (interface_true.compare_features(data=x) and interface_true.compare_target(data=y_pred)):
                LOG.error("Feature or Target names do not match - EXIT")
                sys.exit()
            results.comp_time_pred_train = time.perf_counter() - tic_pred
            LOG.info("Predicted on seen data")
            y_pred.log_status(ignore_log_lvl=True)
            results.y_pred_train_fpath = store_data(f_idend=para.data_identifier, data=y_pred, side="pred", mode="Train")

    # Get Testing Data
    if para.idxs_test is None:
        LOG.info("No Testing Data")
    else:
        LOG.info("Get Testing Data X")
        x = Data()
        x.set_from_files(
            file_paths=para.file_paths_ai_in,
            percentiles=para.feature_percentiles,
            idxs=para.idxs_test,
            columns=para.used_columns_ai_in,
        )
        LOG.info("Got Testing Data X")
        x.log_status(ignore_log_lvl=True)

        LOG.info("Get Testing Data Y")
        y_true = Data()
        y_true.set_from_files(
            file_paths=para.file_paths_ai_out,
            percentiles=para.target_percentiles,
            idxs=para.idxs_test,
            columns=para.used_columns_ai_out,
        )
        LOG.info("Got Testing Data Y")
        y_true.log_status(ignore_log_lvl=True)
        results.y_true_test_fpath = store_data(f_idend=para.data_identifier, data=y_true, side="true", mode="Test")

        # Predict on unseen data
        LOG.info("Predict on unseen data")
        tic_pred = time.perf_counter()
        y_pred = pipe.predict(x=x)
        if not (interface_true.compare_features(data=x) and interface_true.compare_target(data=y_pred)):
            LOG.error("Feature or Target names do not match - EXIT")
            sys.exit()
        results.comp_time_pred_test = time.perf_counter() - tic_pred
        LOG.info("Predicted on unseen data")
        y_pred.log_status(ignore_log_lvl=True)
        results.y_pred_test_fpath = store_data(f_idend=para.data_identifier, data=y_pred, side="pred", mode="Test")

    # get hyperparameters
    results.hyperparameter = pipe.get_params()

    # return
    r_path = Path(f"results_{para.data_identifier}.pkl")
    LOG.info("Store Results in %s", r_path)
    with open(r_path, "wb") as f:
        pickle.dump(results, f)

    LOG.info("Done Fold %s", para.data_identifier)

    return str(r_path)


class Evaluate:
    def __init__(
        self,
        case_ids: List[int],
        pipe: Type[BasePipe],
        pipe_paras: dict,
        file_paths_ai_in: List[Path],
        file_paths_ai_out: List[Path],
        feature_percentiles: Optional[List[int]] = None,
        target_percentiles: Optional[List[int]] = None,
        used_columns_ai_in: Optional[List[str]] = None,
        used_columns_ai_out: Optional[List[str]] = None,
        random_state: Optional[int] = 42,
        shuffle: bool = True,
    ) -> None:
        """Evaluate with n fold cross validation

        Args:
            case_ids (List[int]): IDs to be used
            pipe (Type[BasePipe]): user pipeline
            pipe_paras (dict): parameters of user pipeline
            file_paths_ai_in (List[Path]): file paths of input data
            file_paths_ai_out (List[Path]): file paths of output data
            feature_percentiles (Optional[List[int]], optional): percentiles to select on input side, None if DOE factors. Defaults to None.
            target_percentiles (Optional[List[int]], optional): percentiles to select on output side, None if DOE factors. Defaults to None.
            random_state (Optional[int], optional): random state of KFOLD. Defaults to 42.
            shuffle (bool, optional): controls KFOLD shuffle. Defaults to True.
        """
        self.str = StandardNames()

        # dev set
        self.file_paths_ai_in: List[Path] = file_paths_ai_in
        self.file_paths_ai_out: List[Path] = file_paths_ai_out
        self.feature_percentiles: Optional[List[int]] = feature_percentiles
        self.target_percentiles: Optional[List[int]] = target_percentiles
        self.used_columns_ai_in: Optional[List[str]] = used_columns_ai_in
        self.used_columns_ai_out: Optional[List[str]] = used_columns_ai_out
        self.case_ids: List[int] = case_ids

        # pipeline
        self.pipe: Type[BasePipe] = ObjectChecker().pipeline(pipe=pipe)
        self.pipe_paras: Dict[str, Union[str, int, float, bool]] = pipe_paras
        self.pipes: List[BasePipe] = []

        # process
        self.shuffle: bool = shuffle
        self.random_state: Optional[int] = random_state
        self.n_splits: int = 5

        # cross fold validation
        self.fold_idxs: List[Tuple[np.ndarray, np.ndarray]] = []

        # evaluation
        n_anthros = (len(self.feature_percentiles) if self.feature_percentiles is not None else 0) + (
            len(self.target_percentiles) if self.target_percentiles is not None else 0
        )
        self.results: ResultsGlobal = ResultsGlobal(
            n_crash_simulations=1,  # all pulses scaled from single crash
            n_occupant_simulations=len(self.case_ids) * n_anthros,
            n_target_anthros=len(self.target_percentiles) if self.target_percentiles is not None else 0,
            frac_train=1,
            frac_test=0,  # cross fold validation uses all data for training and testing
        )

        # fill mcdm
        self.results.mcdm_criteria.add_setup_db_num_sim_calibration_pprediction(
            n_calibrators=len(self.feature_percentiles) if self.feature_percentiles is not None else 0
        )
        self.results.mcdm_criteria.add_setup_db_num_sim_calibration_sum(n_vehicle_environments=len(self.case_ids))

        self.results.mcdm_criteria.add_setup_db_num_sim_crash()
        self.results.mcdm_criteria.add_setup_db_comp_time_crash()
        self.results.mcdm_criteria.add_setup_db_num_sim_occupant()
        self.results.mcdm_criteria.add_setup_db_num_sim_training()
        LOG.warning("Not implemented: Add MCDM Criteria setup_db_num_sim_assessment - constant values")
        self.results.mcdm_criteria.add_setup_db_num_sim_assessment(
            n_samples_total_assessment=0
        )  # cross fold uses all data for training and testing, no extrapolation implemented
        self.results.mcdm_criteria.add_setup_training_num_sim_occupant()
        self.results.mcdm_criteria.add_setup_training_num_sim_crash()
        self.results.mcdm_criteria.add_setup_training_comp_time_crash()
        self.results.mcdm_criteria.add_setup_training_calibration_sum()
        self.results.mcdm_criteria.add_setup_test_num_sim_occupant()
        self.results.mcdm_criteria.add_setup_test_num_sim_crash()
        self.results.mcdm_criteria.add_setup_test_comp_time_crash()
        self.results.mcdm_criteria.add_setup_test_calibration_sum()
        self.results.mcdm_criteria.add_us_prediction_sensor_anthropometrics(
            n_target_anthros=len(self.target_percentiles) if self.target_percentiles is not None else 0
        )

    def run(self):
        """Run evaluation process of estimator"""
        LOG.info("Split")
        self.__generate_fold_idxs()
        LOG.info("Train and Test")
        result_fpaths = self.__train_test()
        self.__assess(result_fpaths=result_fpaths)

    def __generate_fold_idxs(self):
        """Generate indices for folds"""
        LOG.info("Split Dev Set")
        # shuffle
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        case_ids = np.array(self.case_ids)
        self.fold_idxs = [(case_ids[train_idx], case_ids[test_idx]) for train_idx, test_idx in kf.split(self.case_ids)]

        # document
        self.results.add_data_split_info(
            split_paras=kf.__dict__,
            split_type=str(type(kf)),
            fold_info={i: {self.str.train: len(fold[0]), self.str.test: len(fold[1])} for i, fold in enumerate(self.fold_idxs)},
        )

        LOG.info("Generated %s folds", len(self.fold_idxs))

    def __train_test(self) -> List[str]:
        """Fit estimators"""
        # prepare
        iterable = [
            TrainTestParaContainer(
                pipeline=self.pipe,
                data_identifier=fold_id if fold_id < len(self.fold_idxs) else -1,
                idxs_train=train_idx,
                idxs_test=test_idx,
                file_paths_ai_in=self.file_paths_ai_in,
                file_paths_ai_out=self.file_paths_ai_out,
                feature_percentiles=self.feature_percentiles,
                target_percentiles=self.target_percentiles,
                used_columns_ai_in=self.used_columns_ai_in,
                used_columns_ai_out=self.used_columns_ai_out,
                pipe_paras=self.pipe_paras,
            )
            for fold_id, (train_idx, test_idx) in enumerate(self.fold_idxs + [(self.case_ids, None)])
        ]

        LOG.info("Start Training and Testing - Spawn %s Processes", len(iterable))
        if self.pipe()._run_in_subprocess:
            LOG.info("Run in Subprocess")
            with multiprocessing.Pool(processes=1, maxtasksperchild=1) as pool:
                result_fpaths: List[str] = pool.map(func=train_test_parallel, iterable=iterable, chunksize=10)
        else:
            LOG.info("Run in Main Process")
            result_fpaths: List[str] = [train_test_parallel(para=p) for p in iterable]
        LOG.info("Done Training and Testing")

        return result_fpaths

    def __assess(self, result_fpaths: List[str]) -> None:
        n_classes = (
            int(self.file_paths_ai_out[0].stem.split("_")[-1])
            if self.file_paths_ai_out[0].stem.startswith("injury_criteria_classes_")
            else None
        )
        LOG.info("Load Results")
        y_true_conc, y_pred_conc = Data(), Data()
        for p_path in result_fpaths:
            y_true, y_pred = self.__assess_result(p_path=Path(p_path), n_classes=n_classes)
            if y_true is not None and y_pred is not None:
                y_true_conc.append(y_true)
                y_pred_conc.append(y_pred)

        # get full test score
        LOG.info("Score on full test set")
        self.results.update_scores(
            scores=score(
                y_true=y_true_conc,
                y_pred=y_pred_conc,
                fold_id=-1,
                mode="Test",
                n_classes=n_classes,
            )
        )
        LOG.info("Done Full Test Score")

        LOG.info("Add MCDM Criteria")
        self.__add_mcdm_criteria(y_pred=y_pred_conc, n_classes=n_classes)
        LOG.info("Done MCDM Criteria")

    def __assess_result(self, p_path: Path, n_classes: int) -> Tuple[Data, Data]:
        # read result
        LOG.info("Load Results from %s", p_path)
        with open(p_path, "rb") as f:
            result: Results = pickle.load(f)
        p_path.unlink()
        LOG.debug("Deleted file %s from %s: %s", p_path, Path().absolute(), p_path.is_file())

        # eval / store
        self.results.update_local_info(results=result)

        # scoring train
        if result.y_true_train_fpath is not None and result.y_pred_train_fpath is not None:
            LOG.info("Score Training of %s", p_path)
            self.results.update_scores(
                scores=score(
                    y_true=read_data(result.y_true_train_fpath),
                    y_pred=read_data(result.y_pred_train_fpath),
                    fold_id=result.fold_id,
                    mode="Train",
                    n_classes=n_classes,
                )
            )
        else:
            LOG.warning("No Training Data in %s", p_path)

        # scoring test
        if result.y_true_test_fpath is not None and result.y_pred_test_fpath is not None:
            LOG.info("Score Testing of %s", p_path)
            y_true, y_pred = read_data(result.y_true_test_fpath), read_data(result.y_pred_test_fpath)
            self.results.update_scores(
                scores=score(
                    y_true=y_true,
                    y_pred=y_pred,
                    fold_id=result.fold_id,
                    mode="Test",
                    n_classes=n_classes,
                )
            )
        else:
            LOG.info("No Testing Data in %s", p_path)
            y_true, y_pred = None, None

        # cleanup
        if LOG.getEffectiveLevel() > logging.DEBUG:
            LOG.info("Delete Target Files")
            if result.y_true_train_fpath is not None:
                result.y_true_train_fpath.unlink(missing_ok=True)
            if result.y_pred_train_fpath is not None:
                result.y_pred_train_fpath.unlink(missing_ok=True)
            if result.y_true_test_fpath is not None:
                result.y_true_test_fpath.unlink(missing_ok=True)
            if result.y_pred_test_fpath is not None:
                result.y_pred_test_fpath.unlink(missing_ok=True)
            LOG.debug(
                "Deleted files %s, %s, %s, %s",
                result.y_true_train_fpath,
                result.y_pred_train_fpath,
                result.y_true_test_fpath,
                result.y_pred_test_fpath,
            )

        return y_true, y_pred

    def __add_mcdm_criteria(self, y_pred: Data, n_classes: Optional[int] = None) -> None:
        # setup cost training
        self.results.mcdm_criteria.add_setup_training_comp_time_assessment_sum(
            t_elapsed=sum(self.results.data_info[f]["comp_time_pred_train"] for f in self.results.data_info.keys())
        )
        self.results.mcdm_criteria.add_setup_training_comp_time_assessment_pprediction(
            t_elapsed=self.results.data_info[-1]["comp_time_pred_train"] / len(self.case_ids)
        )
        self.results.mcdm_criteria.add_setup_training_comp_time_calibration()
        self.results.mcdm_criteria.add_setup_training_comp_time_metamodel(
            t_elapsed=sum(self.results.data_info[f]["comp_time_fit"] for f in self.results.data_info.keys())
        )

        # setup cost test
        self.results.mcdm_criteria.add_setup_test_comp_time_assessment_sum(
            t_elapsed=sum(self.results.data_info[f]["comp_time_pred_test"] for f in range(5))
        )
        self.results.mcdm_criteria.add_setup_test_comp_time_assessment_pprediction(
            t_elapsed=self.results.data_info[-1]["comp_time_pred_train"] / len(self.case_ids)
        )

        # setup cost extrapolation assessment
        LOG.warning(
            "Not implemented: Add MCDM Criteria setup_val_comp_time_assessment_pprediction and setup_val_comp_time_assessment_sum - constant values"
        )
        self.results.mcdm_criteria.add_setup_val_comp_time_assessment_pprediction(t_elapsed=0)  # not implemented
        self.results.mcdm_criteria.add_setup_val_comp_time_assessment_sum(t_elapsed=0)  # not implemented

        # usage
        self.results.mcdm_criteria.add_us_metamodel_time_prediction()
        LOG.info("Got Metric table:\n%s", self.results.get_scores())
        res = self.results.get_scores()
        if len(res.index.names) == 2:
            sc = res.loc[-1, "Test"].mean()
        else:
            sc = res.loc[(slice(None), -1, "Test"), :].mean().mean()
        self.results.mcdm_criteria.add_us_ml_metric(score=sc)

        # add prediction type
        if y_pred.get_temporal() is not None:
            # Full sensor time series
            self.results.mcdm_criteria.add_us_prediction_outputs(degree=1)
        else:
            if y_pred.get_tabular().shape[1] == 1:
                # Single value
                self.results.mcdm_criteria.add_us_prediction_outputs(degree=3)
            else:
                # Relevant characteristics
                self.results.mcdm_criteria.add_us_prediction_outputs(degree=2)

        # add prediction regions
        if y_pred.get_temporal() is not None:
            sensors_used = set(y_pred.get_temporal().columns)
            self.results.mcdm_criteria.add_us_prediction_sensor_num(n_sensors_used=len(sensors_used))
        else:
            sensors_used = set([])
            for inj_vals in y_pred.get_tabular().columns:
                if inj_vals in self.str.inj2channels:
                    sensors_used |= set([ch for ch in self.str.inj2channels[inj_vals]])
                else:
                    LOG.error("Injury value %s not found in inj2channels", inj_vals)
            self.results.mcdm_criteria.add_us_prediction_sensor_num(n_sensors_used=len(sensors_used))
        locations_used = set([ch[2:6] for ch in sensors_used])
        self.results.mcdm_criteria.add_us_prediction_sensor_relevance(sensor_locations=list(locations_used))

        # add prediction type
        self.results.mcdm_criteria.add_us_prediction_type(n_classes=n_classes)

        # number of calibrators
        self.results.mcdm_criteria.add_us_metamodel_num_sim_calibration()

        # not implemented
        LOG.warning("Not implemented: Add MCDM Criteria val_range - constant values")
        self.results.mcdm_criteria.add_val_range(width_space=0)
