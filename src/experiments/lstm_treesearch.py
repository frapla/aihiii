import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src._StandardNames import StandardNames
from src.build.LstmFeatureExtractor import LSTMUniversal
from src.experiments._Experiment import Experiment

LOG: logging.Logger = logging.getLogger(__name__)


class Hyperparameter:
    def __init__(self):
        self.ai_in: List[str] = ["channels", "injury_criteria"]
        self.perc: int = 95
        self.dense_layer_shapes: List[int] = [50, 20]
        self.is_bidirectional: bool = False
        self.use_sequence_output: bool = False
        self.share_dense: bool = True
        self.learning_rate: float = 1e-3
        self.lstm_units: List[List[int]] = [[140], [300]]
        self.share_lstm: bool = True
        self.n_stacks: int = 1
        self.dense_layer_shapes: List[int] = [50, 20]
        self.temporal_feature_n_tsps: int = 140
        self.kernel_regularizer: Optional[str] = "l2"
        self.dropout_rate: float = 0.2

    def do_variation(self, hpara_name, variation):
        scores = []

        for val in variation:
            self.__dict__[hpara_name] = val
            scores.append(self.run_and_score(objective=hpara_name))

        self.__dict__[hpara_name] = variation[np.argmax(scores)]

    def run_and_score(
        self,
        objective: str = "base",
        tgt: str = "injury_criteria",
        database: str = "doe_sobol_20240705_194200",
        patience_factor: float = 0.01,
        max_epochs: int = 1000,
        start_early_stopping_from_n_epochs: int = 600,
        baseline_threshold: int = 30,
    ) -> float:
        cwd = os.getcwd()
        work_dir = Path("experiments") / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{objective}"
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
        LOG.info("Working in %s", work_dir)

        exp = Experiment(
            user_pipeline=LSTMUniversal,
            processed_data_dir=Path("..") / ".." / "data" / "doe" / database,
            file_names_ai_in=self.ai_in,
            file_names_ai_out=[tgt],
            feature_percentiles=[50],
            target_percentiles=[self.perc],
            used_columns_ai_out=[
                "Head_HIC15",
                "Head_a3ms",
                "Chest_a3ms",
                "Neck_My_Extension",
                "Neck_Fz_Max_Tension",
                "Neck_Fx_Shear_Max",
                "Chest_Deflection",
                "Femur_Fz_Max_Compression",
                "Chest_VC",
            ],
            used_columns_ai_in=[
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
                "Neck_My_Extension",
                "Neck_My_Flexion",
                "Neck_Fx_Shear_Max",
                "Chest_Deflection",
                "Chest_a3ms",
                "Chest_VC",
                "Femur_Fz_Max_Compression",
                "Femur_Fz_Max_Tension",
                "Femur_Fz_Max",
            ],
            hyperparameter={
                "dense_layer_shapes": self.dense_layer_shapes,
                "temporal_feature_n_tsps": self.temporal_feature_n_tsps,
                "is_bidirectional": self.is_bidirectional,
                "use_sequence_output": self.use_sequence_output,
                "share_dense": self.share_dense,
                "learning_rate": self.learning_rate,
                "share_lstm": self.share_lstm,
                "dropout_rate": self.dropout_rate,
                "kernel_regularizer": self.kernel_regularizer,
                "lstm_units": [lstm_unit * self.n_stacks for lstm_unit in self.lstm_units],
                "patience_factor": patience_factor,
                "max_epochs": max_epochs,
                "start_early_stopping_from_n_epochs": start_early_stopping_from_n_epochs,
                "baseline_threshold": baseline_threshold,
                "feature_extractor_path": None,
                "plot_model": True,
            },
            shuffle_data=True,
            random_state_shuffle=42,
        )
        LOG.info("Prepare and run experiment")
        exp.prepare()
        exp.run()
        score = exp.get_score()

        os.chdir(cwd)

        LOG.info("Done, back in %s", cwd)

        return score


def test() -> None:
    # init
    hyper_para = Hyperparameter()

    if False:
        # basics
        hyper_para.do_variation("ai_in", [["channels", "injury_criteria"], ["channels"]])
        hyper_para.do_variation("temporal_feature_n_tsps", [140, 70])
        hyper_para.do_variation("learning_rate", [1e-3, 1e-4, 1e-5])
        hyper_para.do_variation("share_dense", [True, False])
        hyper_para.do_variation("share_lstm", [True, False])

        # lstm shape
        hyper_para.do_variation("lstm_units", [[[140], [300]], [[100]], [[140], [280], [70]]])
        hyper_para.do_variation("n_stacks", [1, 3, 6])
        hyper_para.do_variation("is_bidirectional", [True, False])
        hyper_para.do_variation("use_sequence_output", [True, False])

        # dense shape
        hyper_para.do_variation("dense_layer_shapes", [[50, 20], [100, 50, 25], [600, 300, 150, 75], [200, 50]])

        # regularization
        hyper_para.do_variation("dropout_rate", [0.2, 0.5, 0.8])
        hyper_para.do_variation("kernel_regularizer", ["l2", None])
    else:
        hyper_para.ai_in = ["channels", "injury_criteria"]
        hyper_para.temporal_feature_n_tsps = 140
        hyper_para.learning_rate = 1e-5
        hyper_para.share_dense = True
        hyper_para.share_lstm = True
        hyper_para.lstm_units = [[140, 280, 70]]
        hyper_para.n_stacks = 1
        hyper_para.is_bidirectional = False
        hyper_para.use_sequence_output = True
        hyper_para.dense_layer_shapes = [100, 96, 93, 90]
        hyper_para.dropout_rate = 0.2
        hyper_para.kernel_regularizer = "l2"

    # percentiles
    hyper_para.do_variation("perc", [5, 95])


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    test()
