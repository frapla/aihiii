#!/mnt/cluster/home/plaschkies/miniconda3/bin/python
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
import pandas as pd
import os
from pathlib import Path
import numpy as np

b_path = Path("/mnt/cluster/home/plaschkies/sobol")
os.chdir(b_path)
channels_fpath = b_path / "channels.parquet"

relevant_columns = [
            "03CHST0000OCCUACXD",
            "03CHST0000OCCUACYD",
            "03CHST0000OCCUACZD",
            "03CHST0000OCCUDSXD",
            "03CHSTLOC0OCCUDSXD",
            "03CHSTLOC0OCCUDSYD",
            "03CHSTLOC0OCCUDSZD",
            "03FEMRLE00OCCUFOZD",
            "03FEMRRI00OCCUFOZD",
            "03HEAD0000OCCUACXD",
            "03HEAD0000OCCUACYD",
            "03HEAD0000OCCUACZD",
            "03HEADLOC0OCCUDSXD",
            "03HEADLOC0OCCUDSYD",
            "03HEADLOC0OCCUDSZD",
            "03NECKUP00OCCUFOXD",
            "03NECKUP00OCCUFOYD",
            "03NECKUP00OCCUFOZD",
            "03NECKUP00OCCUMOYD",
            "03PELV0000OCCUACXD",
            "03PELV0000OCCUACYD",
            "03PELV0000OCCUACZD",
            "03PELVLOC0OCCUDSXD",
            "03PELVLOC0OCCUDSYD",
            "03PELVLOC0OCCUDSZD",
    ]
    
print("Read", channels_fpath)
db = pd.read_parquet(channels_fpath, columns=relevant_columns, filters=[("PERC", "==", 50)]).apply(pd.to_numeric, downcast="float").droplevel("PERC")
db = np.array(np.split(db.values, db.index.get_level_values("TIME").nunique(), axis=0)).transpose(1,2, 0)

print("Extract")
ts_eff = TSFreshFeatureExtractor(default_fc_parameters="comprehensive", disable_progressbar=True, n_jobs=-1) 
extracted_features = ts_eff.fit_transform(db)
print(extracted_features.shape)

print("Clean")
extracted_features.dropna(axis=1, inplace=True)
standard_devs = extracted_features.std()
extracted_features.drop(columns=list(standard_devs[standard_devs.le(1e-6)].index), inplace=True)
print(extracted_features.shape)

print("Store")
extracted_features.index.name = "ID"
extracted_features["PERC"] = 50
extracted_features.set_index("PERC", append=True, inplace=True)
extracted_features.to_parquet(b_path / "tsfresh_features_50.parquet", index=True)
print("DONE")

