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
train_columns = scaled_train_X.columns.values
