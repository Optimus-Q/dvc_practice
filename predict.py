import re
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from datapipeline import DataPipeline

pattern = re.compile(
    r"(?P<log_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s-\sINFO\s-\s"
    r"Optuna Training Summary\s\|\s"
    r"Start:\s(?P<start_time>[\d\-\s:]+)\s\|\s"
    r"End:\s(?P<end_time>[\d\-\s:]+)\s\|\s"
    r"OPTUNA ACCURACY:\s(?P<optuna_acc>[0-9.]+)\s\|\s"
    r"TEST ACCURACY:\s(?P<test_acc>[0-9.]+)\s\|\s"
    r"Model:\s(?P<model_name>.+)"
)

# system args
test_yamlfile_loc = sys.argv[1]

# import yaml configs
with open(test_yamlfile_loc, "r") as yf:
    test_yamlfile = yaml.safe_load(yf)

# yaml file configs
TESTDATALOC = test_yamlfile['test data']
FEATURESELECTED = test_yamlfile['features selected']
DATAMAPLOC = test_yamlfile["data mapper"]
MODELLOGS = test_yamlfile["model logs"]
MODELLOC = test_yamlfile["model loc"]
SAVERESULT = test_yamlfile["save result"]
KAGGLESUBMISSION = test_yamlfile["kaggle submission"]

# test data
test_df = pd.read_csv(TESTDATALOC)
test_df = test_df[FEATURESELECTED]
test_df = DataPipeline(test_df, DATAMAPLOC)
test_arr = np.array(test_df)

# log data
records = []
with open(MODELLOGS, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            records.append(match.groupdict())

# df logs
df = pd.DataFrame(records)
df.sort_values(by="test_acc", ascending=False)["model_name"].iloc[0]

# best model
best_trained_model = df.sort_values(by="test_acc", ascending=False)["model_name"].iloc[0]
model_bestname = best_trained_model.split("_")[0]
if model_bestname == "rf":
    model_loc = MODELLOC+best_trained_model+".pkl"
    model_ = joblib.load(model_loc)
    pred_ = model_.predict(test_arr)
elif model_bestname == "lgbm":
    model_loc = MODELLOC+best_trained_model+".json"
    model_ = lgb.Booster(model_file=model_loc)
    pred_ = model_.predict(test_arr)


# save results
time_format = "%Y-%m-%d %H:%M:%S"
test_time = datetime.now().strftime(format=time_format).replace(':', '-')
result_filename = f"submission_{test_time}.csv"
df_sub = pd.read_csv(KAGGLESUBMISSION)
df_sub["Predict"] = [int(x) for x in pred_.tolist()]
df_sub.to_csv(SAVERESULT+result_filename, index=False)