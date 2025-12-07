# Import libraries
import os
import sys
import logging
import optuna
import json
import joblib
import yaml
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from datetime import datetime
from datapipeline import DataPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=UserWarning)


# data function
def getdata(traindfloc, featsel, maploc, traindatasz, trainrandstate, trainshuffle):
    train_df = pd.read_csv(traindfloc)
    train_df = train_df[featsel]
    train_data = DataPipeline(train_df, maploc)
    x = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    train_size = traindatasz, 
                                                    random_state = trainrandstate, 
                                                    shuffle = trainshuffle, stratify = y)
    xtrain_arr = np.array(xtrain)
    xtest_arr = np.array(xtest)
    ytrain_arr = np.array(ytrain)
    ytest_arr = np.array(ytest)
    return (xtrain_arr, xtest_arr, ytrain_arr, ytest_arr)

# training objective function -> optuna
def objective(trial, modelname):
    accuracy_folds = []
    cv_ = StratifiedKFold(n_splits = KFOLDSPLIT, shuffle = TRAINSHUFFLE, random_state = TRAINRANDOMSTATE)
    for fold, (train_index, valid_index) in enumerate(cv_.split(xtrainArr, ytrainArr)):
        xtrain_arr_fold, xvalid_arr_fold = xtrainArr[train_index], xtrainArr[valid_index]
        ytrain_arr_fold, yvalid_arr_fold = ytrainArr[train_index], ytrainArr[valid_index]
        if modelname == "rf":
            _params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_jobs": -1,
                "random_state": TRAINRANDOMSTATE}
            model_rf = RandomForestClassifier(**_params).fit(xtrain_arr_fold, ytrain_arr_fold)
            pred_rf = model_rf.predict(xvalid_arr_fold)
            metric_rf = accuracy_score(yvalid_arr_fold, pred_rf)
            accuracy_folds.append(metric_rf)
        elif modelname == "lgbm":
            _params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 256),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": TRAINRANDOMSTATE,
                "n_jobs": -1,
                "verbose": -1,
                "verbosity": -1}
            model_lgb = lgb.LGBMClassifier(**_params).fit(xtrain_arr_fold, ytrain_arr_fold)
            pred_lgb = model_lgb.predict(xvalid_arr_fold)
            metric_lgb = accuracy_score(yvalid_arr_fold, pred_lgb)
            accuracy_folds.append(metric_lgb)
    agg_metric = np.mean(accuracy_folds)
    try:
        best_so_far = trial.study.best_value
    except ValueError:
        best_so_far = None
    print(
        f"Trial {trial.number} | "
        f"Current Score: {agg_metric:.5f} | "
        f"Best Score So Far: {best_so_far}"
    )
    return agg_metric

# test
def testmodel(optimisedmodel, modelname):
    if modelname == "rf":
        best_params = optimisedmodel.best_params
        best_params.update({"random_state": TRAINRANDOMSTATE})
        model_ = RandomForestClassifier(**best_params).fit(xtrainArr, ytrainArr)
        pred_ = model_.predict(xtestArr)
    elif modelname == "lgbm":
        best_params = optimisedmodel.best_params
        best_params.update({"random_state": TRAINRANDOMSTATE})
        model_ = lgb.LGBMClassifier(**best_params).fit(xtrainArr, ytrainArr)
        pred_ = model_.predict(xtestArr)
    return model_, pred_

# logs
logging.basicConfig(
    filename="./train/train logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# system args
train_yamlfile_loc = sys.argv[1]

# import yaml configs
with open(train_yamlfile_loc, "r") as yf:
    train_yamlfile = yaml.safe_load(yf)

# yaml file configs
TRAINDATALOC = train_yamlfile['train data']
FEATURESELECTED = train_yamlfile['features selected']
FEATURES_SELECTED_TRAIN = FEATURESELECTED + ['Label']
DATAMAPLOC = train_yamlfile["data mapper"]
TRAINDATASIZE = train_yamlfile["train data size"]
TRAINRANDOMSTATE = train_yamlfile["train random state"]
TRAINSHUFFLE = train_yamlfile["train shuffle"]
KFOLDSPLIT = 7
MODELNAME = train_yamlfile["model name"]
EPCOHS = train_yamlfile["epochs"]

# process and map data
xtrainArr, xtestArr, ytrainArr, ytestArr = getdata(traindfloc=TRAINDATALOC, featsel=FEATURES_SELECTED_TRAIN,
                                                   maploc=DATAMAPLOC, traindatasz=TRAINDATASIZE,
                                                   trainrandstate=TRAINRANDOMSTATE, trainshuffle=TRAINSHUFFLE)

# optimisation
study = optuna.create_study(direction="maximize")
study.optimize(
    lambda trial: objective(
        trial,
        modelname=MODELNAME),
    n_trials=EPCOHS
)

# test model
optim_models, pred_test = testmodel(optimisedmodel=study, modelname=MODELNAME)

# study details
time_format = "%Y-%m-%d %H:%M:%S"
study_accuracy_value = study.best_trial.values[0]
study_start_time = study.best_trial.datetime_start.strftime(format=time_format)
study_end_time = study.best_trial.datetime_complete.strftime(format=time_format)
study_model_filename = f"{MODELNAME}_train_model_{study_end_time.replace(':', '-')}"
test_accuracy_score = accuracy_score(ytestArr, pred_test)

# log details
logging.info(
    "Optuna Training Summary | Start: %s | End: %s | OPTUNA ACCURACY: %.4f | TEST ACCURACY: %.4f | Model: %s",
    study_start_time,
    study_end_time,
    study_accuracy_value,
    test_accuracy_score,
    study_model_filename)

# save model
if MODELNAME == "rf":
    joblib.dump(optim_models, f"./train/model weights/{study_model_filename}.pkl")
elif MODELNAME == "lgbm":
    optim_models.booster_.save_model(f"./train/model weights/{study_model_filename}.json")