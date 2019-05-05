# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:17:51 2019

@author: 29132
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from def_functions import *
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#%%
scores=load_data('赛题1数据集/企业评分.xlsx')
train_remain_bond=pd.read_csv('data/train_remain_bond0422.csv')
train_remain_finance=pd.read_csv('data/train_remain_finance0422.csv')
train_remain_land=pd.read_csv('data/train_remain_land0422.csv')
train_remain_credit=pd.read_csv('data/train_remain_credit0422.csv')
train_remain_compete=pd.read_csv('data/train_remain_compete0422.csv')
test_remain_bond=pd.read_csv('data/test_remain_bond0422.csv')
test_remain_finance=pd.read_csv('data/test_remain_finance0422.csv')
test_remain_land=pd.read_csv('data/test_remain_land0422.csv')
test_remain_credit=pd.read_csv('data/test_remain_credit0422.csv')
test_remain_compete=pd.read_csv('data/test_remain_compete0422.csv')
remain_bond=pd.concat([train_remain_bond,test_remain_bond]).reset_index(drop=True)
remain_finance=pd.concat([train_remain_finance,test_remain_finance]).reset_index(drop=True)
remain_land=pd.concat([train_remain_land,test_remain_land]).reset_index(drop=True)
remain_credit=pd.concat([train_remain_credit,test_remain_credit]).reset_index(drop=True)
remain_compete=pd.concat([train_remain_compete,test_remain_compete]).reset_index(drop=True)
#%%
drop_values=['半年付息','附息式浮动利率','按季付息']
remain_bond=remain_bond[~remain_bond['付息方式'].isin(drop_values)]
remain_bond=remain_bond.reset_index(drop=True)
remain_bond=pd.get_dummies(remain_bond).groupby('企业编号',as_index=False).sum()
remain_finance=pd.get_dummies(remain_finance).groupby('企业编号',as_index=False).sum()
remain_land=pd.get_dummies(remain_land).groupby('企业编号',as_index=False).sum()
remain_compete=pd.get_dummies(remain_compete).groupby('企业编号',as_index=False).sum()
remain_credit.replace('0000000001',np.nan,inplace=True)
remain_credit['信用等级']=remain_credit['信用等级'].fillna(remain_credit['信用等级'].mode()[0])
remain_credit=pd.get_dummies(remain_credit).groupby('企业编号',as_index=False).sum()
for i in remain_bond.columns[1:]:
    remain_bond[i]=remain_bond[i].apply(convert_values_to_one)
for i in remain_finance.columns[1:]:
    remain_finance[i]=remain_finance[i].apply(convert_values_to_one)
for i in remain_land.columns[1:]:
    remain_land[i]=remain_land[i].apply(convert_values_to_one)
for i in remain_compete.columns[1:]:
    remain_compete[i]=remain_compete[i].apply(convert_values_to_one)
for i in remain_credit.columns[1:]:
    remain_credit[i]=remain_credit[i].apply(convert_values_to_one)
#%%
train_word_features=[remain_credit[remain_credit['企业编号']<4001],remain_bond[remain_bond['企业编号']<4001],\
remain_compete[remain_compete['企业编号']<4001],remain_finance[remain_finance['企业编号']<4001],\
remain_land[remain_land['企业编号']<4001]]
test_word_features=[remain_credit[remain_credit['企业编号']>=4001],remain_bond[remain_bond['企业编号']>=4001],\
remain_compete[remain_compete['企业编号']>=4001],remain_finance[remain_finance['企业编号']>=4001],\
remain_land[remain_land['企业编号']>=4001]]
#%%
data=np.arange(1001,4001)
train_word_data = pd.DataFrame(data,columns=['企业编号'])
for i in range(len(train_word_features)):
    train_word_data=pd.merge(train_word_data,train_word_features[i],on='企业编号',how='outer')
test_word_data=load_data('企业编号.xlsx')
for i in range(len(test_word_features)):
    test_word_data=pd.merge(test_word_data,test_word_features[i],on='企业编号',how='outer')
#%%
train_word_data.to_csv("data/train_word_0422.csv",index=False,na_rep="NULL",encoding='utf_8_sig')
test_word_data.to_csv("data/test_word_0422.csv",index=False,na_rep="NULL",encoding='utf_8_sig')