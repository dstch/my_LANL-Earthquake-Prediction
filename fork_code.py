#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: fork_code.py
@time: 2019/5/26 11:38
@desc: fork from https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

PATH = "../input/"
os.listdir(PATH)

train_df = pd.read_csv(os.path.join(PATH, 'train.csv'),
                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

rows = 150000
segments = int(np.floor(train_df.shape[0] / rows))
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)


params = {'num_leaves': 51,
          'min_data_in_leaf': 10,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "feature_fraction": 0.91,
          "bagging_freq": 1,
          "bagging_fraction": 0.91,
          "bagging_seed": 42,
          "metric": 'mae',
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": -1,
          "random_state": 42}

scaler = StandardScaler()
scaler.fit(train_X)
scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
train_columns = scaled_train_X.columns.values

oof = np.zeros(len(scaled_train_X))
predictions = np.zeros(len(scaled_test_X))
feature_importance_df = pd.DataFrame()
# run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)

    X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators=20000, n_jobs=-1)
    model.fit(X_tr,
              y_tr,
              eval_set=[(X_tr, y_tr), (X_val, y_val)],
              eval_metric='mae',
              verbose=1000,
              early_stopping_rounds=500)
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    # predictions
    predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits

submission.time_to_failure = predictions
submission.to_csv('submission.csv', index=True)
