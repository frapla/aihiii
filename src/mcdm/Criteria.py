import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.mcdm.Names import Names

STR: Names = Names()


class Criteria:
    def __init__(
        self,
        n_crash_simulations: int,
        n_occupant_simulations: int,
        n_target_anthros: int,
        frac_train: float = 0.8,
        frac_test: float = 0.2,
        t_1oc_sim: float = 0 * 60 * 1000,  # 20min per occupant simulation, in ms
        t_1ch_sim: float = 8 * 60 * 60 * 1000,  # 8h per crash simulation TODO check
    ):
        # constants
        self.n_crash_simulations: int = n_crash_simulations
        self.n_occupant_simulations: int = n_occupant_simulations
        self.n_target_anthros: int = n_target_anthros

        # parameter
        self.frac_train: float = frac_train
        self.frac_test: float = frac_test
        self.t_1oc_sim: float = t_1oc_sim
        self.t_1ch_sim: float = t_1ch_sim
        self.relevances: Dict[str, int] = {"CHST": 1, "HEAD": 1, "NECK": 1, "PELV": 3, "FEMR": 1, "FACE": 4, "SHLD": 4, "KNEE": 2}
        self.pred_type_marks: Dict[int, int] = {None: 1, 7: 2, 5: 3, 4: 3, 3: 4, 2: 5}

        # init
        self._assessed_criteria: dict = {}

    def get_assessed_criteria(self):
        return self._assessed_criteria

    def __add_criterion(self, name: str, value: float):
        if name not in self._assessed_criteria:
            self._assessed_criteria[name] = value
        else:
            logging.error(msg=f"Criteria {name} exist already - keep old")

    def __get_referred(self, ref_name: str):
        if ref_name in self._assessed_criteria:
            value = self._assessed_criteria[ref_name]
        else:
            logging.error(msg=f"The criteria {ref_name} has to be referred beforehand!")
            value = np.nan

        return value

    def add_setup_db_num_sim_calibration_pprediction(self, n_calibrators: int):
        """Calibration simulations per environment (= per prediction)

        Args:
            n_calibrators (int): number of calibrators
        """
        self.__add_criterion(name="setup_db_num_sim_calibration_pprediction", value=n_calibrators)

    def add_setup_db_num_sim_calibration_sum(self, n_vehicle_environments: int):
        """Total number of calibration simulations in database
            Requires: setup_db_num_sim_calibration_pprediction

        Args:
            n_vehicle_environments (int): number of vehicle environments
        """
        self.__add_criterion(
            name="setup_db_num_sim_calibration_sum",
            value=n_vehicle_environments * self.__get_referred("setup_db_num_sim_calibration_pprediction"),
        )

    def add_setup_db_num_sim_crash(self):
        """Total number of crash simulations in database

        Args:
            num_sim_crash (int): Total number of crash simulations in database
        """
        self.__add_criterion(name="setup_db_num_sim_crash", value=self.n_crash_simulations)

    def add_setup_db_num_sim_occupant(self):
        """Total number of occupant simulations in database"""
        self.__add_criterion(name="setup_db_num_sim_occupant", value=self.n_occupant_simulations)

    def add_setup_db_num_sim_training(self):
        """Total number of simulations used for metamodel training
        Requires: setup_db_num_sim_crash, setup_db_num_sim_occupant
        """
        self.__add_criterion(
            name="setup_db_num_sim_training",
            value=self.frac_train
            * (self.__get_referred("setup_db_num_sim_crash") + self.__get_referred("setup_db_num_sim_occupant")),
        )

    def add_setup_db_num_sim_assessment(self, n_samples_total_assessment: int):
        """Total number of samples used for metamodel assessment (test & validation)

        Args:
            n_simulations_total (int): Total number of samples used for metamodel assessment
        """
        self.__add_criterion(name="setup_db_num_sim_assessment", value=n_samples_total_assessment)

    def add_setup_db_comp_time_crash(self):
        """Total computation time of all crash simulations in database
        Requires: setup_db_num_sim_crash
        """
        self.__add_criterion(
            name="setup_db_comp_time_crash", value=self.__get_referred("setup_db_num_sim_crash") * self.t_1ch_sim
        )

    def add_setup_training_comp_time_calibration(self):
        """Total computation time for calibration simulations used in training phase
        Requires: add_setup_db_num_sim_calibration_sum
        """
        value = self.frac_train * self.n_target_anthros * self.t_1oc_sim * self.__get_referred("setup_db_num_sim_calibration_sum")

        self.__add_criterion(name="setup_training_comp_time_calibration", value=value)

    def add_setup_training_calibration_sum(self):
        """Number of calibration simulations used  in training phase
        Requires: setup_db_num_sim_calibration_sum
        """
        self.__add_criterion(
            name="setup_training_calibration_sum",
            value=int(self.frac_train * self.__get_referred("setup_db_num_sim_calibration_sum")),
        )

    def add_setup_training_num_sim_crash(self):
        """Total number of crash simulations used in training phase
        Requires: setup_db_num_sim_crash
        """
        self.__add_criterion(
            name="setup_training_num_sim_crash", value=self.frac_train * self.__get_referred("setup_db_num_sim_crash")
        )

    def add_setup_training_num_sim_occupant(self):
        """Total number of occupant simulations used in training phase
        Requires: setup_db_num_sim_occupant
        """
        self.__add_criterion(
            name="setup_training_num_sim_occupant",
            value=self.frac_train * self.__get_referred("setup_db_num_sim_occupant"),
        )

    def add_setup_training_comp_time_crash(self):
        """Total computation time for crash simulations used in training phase
        Requires: setup_training_num_sim_crash
        """
        self.__add_criterion(
            name="setup_training_comp_time_crash",
            value=self.t_1ch_sim * self.__get_referred("setup_training_num_sim_crash"),
        )

    def add_setup_test_calibration_sum(self):
        """Number of calibration simulations used for assessment within trained space
        Requires: setup_db_num_sim_calibration_sum
        """
        self.__add_criterion(
            name="setup_test_calibration_sum",
            value=int(self.frac_test * self.__get_referred("setup_db_num_sim_calibration_sum")),
        )

    def add_setup_test_num_sim_crash(self):
        """Total number of crash simulations used for assessment within trained space
        Requires: setup_db_num_sim_crash
        """
        self.__add_criterion(
            name="setup_test_num_sim_crash", value=self.frac_test * self.__get_referred("setup_db_num_sim_crash")
        )

    def add_setup_test_num_sim_occupant(self):
        """Total number of occupant simulations used for assessment within trained space
        Requires: setup_db_num_sim_occupant
        """
        self.__add_criterion(
            name="setup_test_num_sim_occupant", value=self.frac_test * self.__get_referred("setup_db_num_sim_occupant")
        )

    def add_setup_test_comp_time_crash(self):
        """Total computation time of all crash simulations in database within trained space
        Requires: setup_test_num_sim_crash
        """
        self.__add_criterion(
            name="setup_test_comp_time_crash", value=self.__get_referred("setup_test_num_sim_crash") * self.t_1ch_sim
        )

    def add_us_metamodel_num_sim_calibration(self):
        """Calibration simulations per environment
        Required: setup_db_num_sim_calibration_pprediction
        """
        self.__add_criterion(
            name="us_metamodel_num_sim_calibration", value=self.__get_referred("setup_db_num_sim_calibration_pprediction")
        )

    def add_us_prediction_type(self, n_classes: int):
        """Value of prediction type, grades, where 1 is best
            5 ⇒ Binary classification (e. g. critical, uncritical)
            4 ⇒ 3 classes
            3 ⇒ 4 – 5 classes
            2 ⇒≥ 6 classes
            1 ⇒ Regression

        Args:
            prediction_type (str): name of prediction type
        """
        self.__add_criterion(name="us_prediction_type", value=self.pred_type_marks[n_classes])

    def add_us_prediction_outputs(self, degree: int):
        """Degree of detail of predictions, grades, where 1 is best
        3 ⇒ Single value
        2 ⇒ Relevant characteristics
        1 ⇒ Full sensor time series
        """
        self.__add_criterion(name="us_prediction_outputs", value=degree)

    def add_us_prediction_sensor_num(self, n_sensors_used: int, n_sensors_total: int = 76):
        """Number of not used sensors (reference are available sensors of used dummy)

        Args:
            n_sensors_used (int): number of used sensors
            n_sensors_total (int, optional): number of potential sensors. Defaults to 76.
        """
        self.__add_criterion(name="us_prediction_sensor_num", value=n_sensors_total - n_sensors_used)
        # choose proper sensor num - may change for other dummy model

    def add_us_prediction_sensor_relevance(self, sensor_locations: List[str]):
        """Relevance of used sensors, grades, where 1 is best
            4 ⇒ Irrelevant
            3 ⇒ Physics relevant
            2 ⇒ Utilized in consumer tests
            1 ⇒ Utilized in legislation

        Args:
            sensor_locations (List[str]): locations of used  sensors
        """
        value = float(np.mean([self.relevances[x] for x in sensor_locations]))

        self.__add_criterion(name="us_prediction_sensor_relevance", value=value)

    def add_us_prediction_sensor_anthropometrics(self, n_target_anthros: int):
        """Detail of degree of anthropometrical prediction, grades, where 1 is best
                5 ⇒ 1 percentile
                4 ⇒ 2 percentiles
                3 ⇒ 3 – 4 percentiles
                2 ⇒≥ 5 percentiles
                1 ⇒ Anthropometrical parameter

        Args:
            n_target_anthros (int): number of target anthropometrics
        """
        value = 6 - n_target_anthros
        # mark 1 & 2 not possible yet

        self.__add_criterion(name="us_prediction_anthropometrics", value=value)

    def add_us_metamodel_time_prediction(self):
        """Computation time for single prediction by metamodel in test phase
        Requires: setup_test_comp_time_assessment_pprediction, setup_training_comp_time_assessment_pprediction
        """
        value = float(
            np.mean(
                [
                    self.__get_referred("setup_training_comp_time_assessment_pprediction"),
                    self.__get_referred("setup_test_comp_time_assessment_pprediction"),
                ]
            )
        )

        self.__add_criterion(name="us_metamodel_time_prediction", value=value)

    def add_setup_training_comp_time_metamodel(self, t_elapsed: float):
        """Computation time in training phase

        Args:
            t_elapsed (float): Computation time of metamodel’s training
        """
        self.__add_criterion(name="setup_training_comp_time_metamodel", value=t_elapsed)

    def add_setup_training_comp_time_assessment_sum(self, t_elapsed: float):
        """Computation time of metamodel’s assessment (predictions) on training data in training phase

        Args:
            t_elapsed (float): Computation time of metamodel’s assessment (predictions) on training data
        """
        self.__add_criterion(name="setup_training_comp_time_assessment_sum", value=t_elapsed)

    def add_setup_training_comp_time_assessment_pprediction(self, t_elapsed: float):
        """Computation time for single prediction by metamodel in training phase
        Args:
            t_elapsed (float): Computation time for single prediction by metamodel in training phase
        """
        self.__add_criterion(
            name="setup_training_comp_time_assessment_pprediction",
            value=t_elapsed,
        )

    def add_setup_test_comp_time_assessment_sum(self, t_elapsed: float):
        """Computation time of metamodel’s assessment (predictions) within trained space

        Args:
            t_elapsed (float): Computation time of metamodel’s assessment (predictions)
        """
        self.__add_criterion(name="setup_test_comp_time_assessment_sum", value=t_elapsed)

    def add_setup_test_comp_time_assessment_pprediction(self, t_elapsed: float):
        """Computation time for single prediction by metamodel in test phase within trained space

        Args:
            t_elapsed (float): computation time for single prediction by metamodel in test phase
        """
        self.__add_criterion(
            name="setup_test_comp_time_assessment_pprediction",
            value=t_elapsed,
        )

    def add_setup_val_comp_time_assessment_sum(self, t_elapsed: float):
        """Total computation time for predictions in extrapolation space

        Args:
            t_elapsed (float): computation time for predictions in extrapolation range
        """
        self.__add_criterion(name="setup_val_comp_time_assessment_sum", value=t_elapsed)

    def add_setup_val_comp_time_assessment_pprediction(self, t_elapsed: float):
        """Computation time for single prediction by metamodel  in extrapolation space

        Args:
            num_sim (int): number of samples
            t_elapsed (float): computation time for single prediction by metamodel in validation phase
        """
        self.__add_criterion(name="setup_val_comp_time_assessment_pprediction", value=t_elapsed)

    def add_us_ml_metric(self, score: float):
        """Value from assessed metric; F-score for classification / R2-score for regression

        Args:
            score (dict): metric's value
        """
        self.__add_criterion(name="us_MLmetric", value=1 - score)

    def add_val_range(self, width_space: float):
        """Width of validity range

        Args:
            width_score (_type_): _description_
        """
        self.__add_criterion(name="val_range", value=-1 * width_space)
