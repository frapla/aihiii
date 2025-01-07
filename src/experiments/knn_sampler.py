import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils.custom_log as custom_log
from src.build.AnnUniversalImportableFTExtractor import AnnUniversal
from src.sampling.KnnSampler import KnnSampler

LOG: logging.Logger = logging.getLogger(__name__)


def main(existing_dir: Optional[Path] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_percentile", type=int, required=True)
    parser.add_argument("--existing_dir", type=Path, default=None)
    parser.add_argument("--sobol_m", type=int, default=4)
    parser.add_argument("--n_seeds", type=int, required=True)
    parser.add_argument("--n_neighbors", type=int, required=True)
    parser.add_argument("--strategy", type=str, default="max")

    args = parser.parse_args()
    target_percentile = args.target_percentile
    existing_dir = args.existing_dir

    if existing_dir is None:
        w_dir = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{Path(__file__).stem}_{AnnUniversal.__name__}_perc{target_percentile}"
    else:
        w_dir = Path(existing_dir)

    work_dir = Path("experiments") / w_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    sampler = KnnSampler(
        work_dir=work_dir,
        data_dir=Path("data") / "doe" / "doe_sobol_20240705_194200",
        doe_fname="doe_combined.parquet",
        sobol_m=args.sobol_m,
        n_seeds=args.n_seeds,
        n_neighbors=args.n_neighbors,
        strategy=args.strategy,
        # setting and reference from experiments/2024-08-07-19-34-27_pure_cnn_optuna_95HIII_injury_criteria_from_doe_sobol_20240705_194200_ft_channels/trial_103/parameters.json
        experiment_kwargs={
            "user_pipeline": AnnUniversal,
            "processed_data_dir": Path("..") / ".." / ".." / "data" / "doe" / "doe_sobol_20240705_194200",
            "file_names_ai_in": ["channels"],
            "file_names_ai_out": ["injury_criteria"],
            "feature_percentiles": [50],
            "target_percentiles": [target_percentile],
            "used_columns_ai_out": [
                "Chest_Deflection",
                "Chest_VC",
                "Chest_a3ms",
                "Femur_Fz_Max_Compression",
                "Head_HIC15",
                "Head_a3ms",
                "Neck_Fx_Shear_Max",
                "Neck_Fz_Max_Tension",
                "Neck_My_Extension",
            ],
            "used_columns_ai_in": [
                "03CHST0000OCCUACXD",
                "03CHST0000OCCUACZD",
                "03CHST0000OCCUDSXD",
                "03CHSTLOC0OCCUDSXD",
                "03HEAD0000OCCUACZD",
                "03HEADLOC0OCCUDSXD",
                "03HEADLOC0OCCUDSYD",
                "03HEADLOC0OCCUDSZD",
                "03NECKUP00OCCUFOXD",
                "03NECKUP00OCCUFOZD",
                "03NECKUP00OCCUMOYD",
                "03PELV0000OCCUACYD",
                "03PELV0000OCCUACZD",
                "03PELVLOC0OCCUDSXD",
                "03PELVLOC0OCCUDSYD",
                "03PELVLOC0OCCUDSZD",
            ],
            "hyperparameter": {
                "conv_nfilters_and_size": [[[60, 41]], [[90, 41]], [[50, 11]]],
                "dense_layer_shapes": [100, 96, 93, 90],
                "pooling_size": 5,
                "pooling_strategy": "average",
                "temporal_feature_n_tsps": 70,
                "share_dense": False,
                "learning_rate": 1e-5,
                "spatial_dropout_rate": 0.33269266283466437,
                "dense_regularizer": "l2",
                "patience_factor": 0.01,
                "max_epochs": 3000,
                "start_early_stopping_from_n_epochs": 600,
                "baseline_threshold": 30,
                "feature_extractor_path": None,
            },
            "shuffle_data": True,
            "random_state_shuffle": 42,
        },
        score_threshold=1,
        max_iterations=np.inf,
    )
    sampler.run()


if __name__ == "__main__":
    custom_log.init_logger(log_lvl=logging.INFO)
    LOG.info("Start")
    main()
    LOG.info("End")
