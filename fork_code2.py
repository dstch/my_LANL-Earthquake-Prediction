#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: fork_code2.py
@time: 2019/5/31 10:13
@desc: fork from https://www.kaggle.com/tocha4/lanl-master-s-approach
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm_notebook
import datetime
import time
import random
from joblib import Parallel, delayed

import lightgbm as lgb
from tensorflow import keras
from gplearn.genetic import SymbolicRegressor
from catboost import Pool, CatBoostRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import os

PATH = "../input/lanl-master-s-features-creating-0"
print(os.listdir(PATH))

train_X_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_X_features_865.csv")
train_X_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_X_features_865.csv")
y_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_y.csv", index_col=False, header=None)
y_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_y.csv", index_col=False, header=None)

train_X = pd.concat([train_X_0, train_X_1], axis=0)
train_X = train_X.reset_index(drop=True)
print(train_X.shape)
train_X.head()

y = pd.concat([y_0, y_1], axis=0)
y = y.reset_index(drop=True)
y[0].shape

train_y = pd.Series(y[0].values)

test_X = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")
# del X["seg_id"], test_X["seg_id"]


scaler = StandardScaler()
train_columns = train_X.columns

train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
test_X[train_columns] = scaler.transform(test_X[train_columns])

train_columns = train_X.columns
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

oof = np.zeros(len(train_X))
train_score = []
fold_idxs = []
# if PREDICTION:
predictions = np.zeros(len(test_X))

feature_importance_df = pd.DataFrame()
# run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    fold_idxs.append(val_idx)

    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]
    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

    model = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE",
                              boosting_type="Ordered", task_type="GPU")
    model.fit(X_tr,
              y_tr,
              eval_set=[(X_val, y_val)],
              #               eval_metric='mae',
              verbose=2500,
              early_stopping_rounds=500)
    oof[val_idx] = model.predict(X_val)

    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # predictions
    #     if PREDICTION:

    predictions += model.predict(test_X[train_columns]) / folds.n_splits
    train_score.append(model.best_score_['learn']["MAE"])

cv_score = mean_absolute_error(train_y, oof)
print(
    f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}",
    end=" ")

today = str(datetime.date.today())
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

submission["time_to_failure"] = predictions
submission.to_csv(f'CatBoost_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv', index=False)
