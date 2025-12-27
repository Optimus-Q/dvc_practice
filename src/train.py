import os
import yaml
import pandas as pd
import numpy as np
import argparse
import optuna
from datapipeline import DataPipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dvclive import Live

# load training yaml file
def trainconfig(filepath):
    with open(filepath, "r") as yf:
        _yamldata = yaml.safe_load(yf)
    return _yamldata


# get data & preprocess data
def getdata(configparams):
    trainfilepath = configparams["train data"]
    datamapperloc = configparams["data mapper"]
    features_selected = configparams["features selected"] + configparams["label name"]
    df_train = pd.read_csv(trainfilepath)
    df_train = df_train[features_selected]
    train_data = DataPipeline(df_train, datamapperloc)
    return train_data

# split data into train and eval
def datasplit(data, configparams):
    trainsize = configparams["train data size"]
    trainrandstate = configparams["train random state"]
    trainshuffle = configparams["train shuffle"]
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=trainsize, 
                                                    random_state=trainrandstate,
                                                    shuffle=trainshuffle,
                                                    stratify=y)
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    return xtrain, xtest, ytrain, ytest


# optuna objective function
def objective(trial, Xtrain, Ytrain, Xtest, Ytest, configparams):
    _params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_jobs": -1,
                "random_state": configparams["train random state"]}
    model_rf = RandomForestClassifier(**_params).fit(Xtrain, Ytrain)
    pred_rf = model_rf.predict(Xtest)
    metric_rf = accuracy_score(pred_rf, Ytest)
    return metric_rf
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainyaml", default="train.yaml", help="Path to train.yaml")
    args = parser.parse_args()

    train_yamldata = trainconfig(args.trainyaml)
    train_df = getdata(train_yamldata)
    Xtrain, Xtest, Ytrain, Ytest = datasplit(train_df, train_yamldata)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest, configparams=train_yamldata), n_trials=train_yamldata["epochs"])

    study_best_value = study.best_trial.values[0]
    study_best_params = dict(study.best_params)
    study_best_params.update({'n_jobs': -1, 'random_state': train_yamldata["train random state"]})

    with Live(save_dvc_exp=True) as live:
        live.log_metric("acc", study_best_value)
        live.log_params(study_best_params)