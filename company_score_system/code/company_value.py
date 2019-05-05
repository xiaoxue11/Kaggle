# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:41:56 2019

@author: 29132
"""

import numpy as np 
import pandas as pd 
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from def_functions import *
#%%
train_values= pd.read_csv('data/train_values0422.csv')
test_values= pd.read_csv('data/test_values0422.csv')
train_word= pd.read_csv('data/train_word_0422.csv')
test_word= pd.read_csv('data/test_word_0422.csv')
print ("Data is loaded!")
#%%
test_values['实际税率(%)']=test_values['实际税率(%)'].fillna(test_values['实际税率(%)'].mean())
test_values=test_values.fillna(0)
test_values.isnull().sum().max()
#%%
train_word=train_word.fillna(0)
test_word=test_word.fillna(0)
train_word.isnull().sum().max()
#%%
train=pd.merge(train_values,train_word,on='企业编号',how='inner')
test=pd.merge(test_values,test_word,on='企业编号',how='inner')
#%%
train.drop(['企业编号'], axis=1, inplace=True)
test.drop(['企业编号'], axis=1, inplace=True)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
y = train['企业总评分'].reset_index(drop=True)
#%%
train_features = train.drop(['企业总评分'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
final_features = pd.get_dummies(features).reset_index(drop=True)
#%%
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)
overfit = list(overfit)
X = X.drop(overfit, axis=1)
X_sub = X_sub.drop(overfit, axis=1)
#%%
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
#%%
model_lgb =LGBMRegressor(objective='regression', 
                                       num_leaves=20, 
                                       n_estimators=1000,
                                       learning_rate=0.005,
                                       max_bin=200, 
                                       bagging_fraction=0.8,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       max_depth=5,
                                       feature_fraction=0.6,
                                       feature_fraction_seed=7,
                                       min_child_samples=19,
                                       min_child_weight=0.001,
                                       reg_alpha=0.5,
                                       reg_lambda=0.08,
                                       verbose=-1, 
                                       )
score = cv_rmse(model_lgb)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
#%%
lgb_model_full_data = model_lgb.fit(X, y)
submission=pd.read_excel("企业编号.xlsx",set='\n',enconding='utf8')
submission['scores']=np.around(lgb_model_full_data.predict(X_sub))
submission.to_excel("赛题1结果_Discovery_Tour.xlsx",index=False,na_rep="NULL",encoding='utf_8_sig',header=None)