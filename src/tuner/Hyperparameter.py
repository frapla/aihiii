from typing import List, Optional


class Hyperparameter:
    database: Optional[str] = None
    file_names_ai_in: Optional[List[str]] = None
    file_names_ai_out: Optional[List[str]] = None
    feature_percentiles: List[int] = [50]
    target_percentiles: List[int] = [5]
    used_columns_ai_out: Optional[List[str]] = None
    used_columns_ai_in: Optional[List[str]] = None
    estimator_hyperparameter: Optional[dict] = None
    shuffle_data: bool = True
    random_state_shuffle: int = 42
    used_ids_ai: Optional[List[int]] = None
