from pathlib import Path


class StandardNames:
    def __init__(self) -> None:
        # general
        self.input = "Input"
        self.output = "Output"
        self.feature = "Feature"
        self.target_class = "Class"
        self.feature_2d = "Feature2D"
        self.target = "Target"
        self.dir = "Directory"
        self.labels = "Labels"
        self.hash = "Hash"
        self.info = "Info"
        self.info_2d = "Info2D"
        self.path = "Path"
        self.files = "Files"
        self.pipeline = "Pipeline"
        self.para = "Parameters"
        self.python = "Python"
        self.axis = "Axis"
        self.data_tabular = "Data_Tabular"
        self.data_temporal = "Data_Temporal"
        self.feature_extractor = "Feature_Extractor"

        self.creation = "Creation_Time"
        self.estimator = "Estimator"

        # data
        self.perc = "PERC"
        self.data = "Data"
        self.id = "ID"
        self.time = "TIME"
        self.sim_id = "SIM_ID"
        self.samples = "Sample"
        self.channels = "Channel"
        self.channelss = "Channels"
        self.injury_criteria = "Injury Criteria"
        self.tsps = "Time"
        self.signal = "Signal"
        self.rid = "rid"
        self.iso = "ISO18571"
        self.fold = "Fold"
        self.model = "Model"

        # file names
        self.fname_para = "parameters.json"
        self.fname_data_info = "data_info.json"
        self.fname_data_info_2d = "data_info2D.json"
        self.fname_results = "results.json"
        self.fname_results_info = "results_info.json"
        self.fname_results_csv = "results.csv"
        self.fname_feature = "feature.npy"
        self.fname_feature_2d = "feature2D.npy"
        self.fname_target = "target.npy"
        self.fname_pipe_pickle = "pipeline_dev_fit.pkl"
        self.fname_channels = "channels.parquet"
        self.fname_injury_crit = "injury_criteria.parquet"
        self.fname_sample_score = "sample_score.parquet"
        self.parquet = ".parquet"
        self.fname_dropped_ids = "dropped_ids.json"
        self.fname_sim_id_2_id = "sim_id_2_id.parquet"
        self.fname_interface = "interface.json"

        # scores
        self.result = "Result"
        self.metrics = "Metrics"
        self.epoch = "Epoch_Loss"
        self.confusion = "Confusion"
        self.artico = "artico"
        self.f1 = "f1"
        self.recall = "recall"
        self.precision = "precision"
        self.r2 = "r2"
        self.whole = "Whole"
        self.accuracy = "balanced_accuracy"
        self.test_median = "Median"
        self.test_conf_lo = "Confidence_Lower"
        self.test_conf_up = "Confidence_Upper"
        self.training_metrics = "Training_Metrics"
        self.testing_metrics = "Testing_Metrics"
        self.true = "True"
        self.predicted = "Predicted"
        self.test = "Test"
        self.train = "Train"
        self.comp_time = "Training_Comp_Time_Median_s"
        self.dev_comp_time = "Dev_Comp_Time_s"
        self.dev = "Dev_Set"
        
        # sampler
        self.assess_error = "Assess_Error"
        self.assess_sample_ids = "Assess_Sample_IDs"

        # standard directories
        self.dir_raw_data = Path("data") / "raw"
        self.dir_experiments = Path("experiments")
        self.dir_processed_data = Path("..") / ".." / "data" / "processed"

        # optuna
        self.opt_test_sc = "arTIco Test Score Median"
        self.opt_train_sc = "arTIco Train Score Mean"
        self.opt_delta = "Delta Train Test Scores"
        self.opt_test_conf = "arTIco Test Score Confidence Interval"

        # injury and channels
        self.inj2channels = {
            "Chest_Deflection": ["03CHST0000OCCUDSXD"],
            "Femur_Fz_Max_Compression": ["03FEMRRI00OCCUFOZD", "03FEMRLE00OCCUFOZD"],
            "Femur_Fz_Max_Tension": ["03FEMRRI00OCCUFOZD", "03FEMRLE00OCCUFOZD"],
            "Femur_Fz_Max": ["03FEMRRI00OCCUFOZD", "03FEMRLE00OCCUFOZD"],
            "Head_HIC15": ["03HEAD0000OCCUACXD", "03HEAD0000OCCUACYD", "03HEAD0000OCCUACZD", "03HEAD0000OCCUACRD"],
            "Head_HIC36": ["03HEAD0000OCCUACXD", "03HEAD0000OCCUACYD", "03HEAD0000OCCUACZD", "03HEAD0000OCCUACRD"],
            "Head_a3ms": ["03HEAD0000OCCUACXD", "03HEAD0000OCCUACYD", "03HEAD0000OCCUACZD", "03HEAD0000OCCUACRD"],
            "Chest_a3ms": ["03CHST0000OCCUACXD", "03CHST0000OCCUACYD", "03CHST0000OCCUACZD", "03CHST0000OCCUACRD"],
            "Neck_Fx_Shear_Max": ["03NECKUP00OCCUFOXD"],
            "Neck_Fz_Max_Tension": ["03NECKUP00OCCUFOZD"],
            "Neck_Fz_Max_Compression": ["03NECKUP00OCCUFOZD"],
            "Neck_My_Max": ["03NECKUP00OCCUMOYD"],
            "Neck_Nij": ["03NECKUP00OCCUMOYD", "03NECKUP00OCCUFOZD"],
        }

        # mcdm
        self.mcdm = "MCDM"
        self.alternatives = "Alternatives"
        self.criteria = "Criteria"
