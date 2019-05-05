# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:25:21 2019

@author: 29132
"""

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from def_functions import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#%%
train_num=pd.read_csv('data/train_num0422.csv')
test_num=pd.read_csv('data/test_num0422.csv')
#%%缺失值处理
missing_data=missing_values_table(train_num)
drop_columns=missing_data[missing_data['percent of Total Values'] > 0.5]
train_num = train_num.drop(drop_columns.index,1)
test_num = test_num.drop(drop_columns.index,1)
train_num['实际税率(%)'] = train_num['实际税率(%)'].fillna(train_num['实际税率(%)'].mean())
train_num['毛利率(%)'] = train_num['毛利率(%)'].fillna(train_num['毛利率(%)'].mean())
train_num['存货周转天数(天)'] = train_num['存货周转天数(天)'].fillna(train_num['存货周转天数(天)'].mean())
train_num['总资产周转率(次)'] = train_num['总资产周转率(次)'].fillna(train_num['总资产周转率(次)'].mean())
train_num['应收账款周转天数(天)'] = train_num['应收账款周转天数(天)'].fillna(train_num['应收账款周转天数(天)'].mean())
train_num['行业大类（代码）'] = train_num['行业大类（代码）'].fillna(train_num['行业大类（代码）'].mode()[0])
#%%异常值处理
train_num=train_num[train_num['商标状态']<1500]
train_num=train_num[train_num['证书名称']<2000]
train_num=train_num[train_num['专利类型']<10000]
train_num=train_num[train_num['状态']<1250]
train_num=train_num[train_num['产品类型']<6]
train_num=train_num[train_num['年报年份_y']<6]
train_num=train_num[train_num['每股未分配利润(元)']<200]
train_num=train_num[train_num['每股净资产(元)']<150]
train_num=train_num[train_num['营业总收入(元)']<40000]
train_num['营业总收入(元)']=np.log(train_num['营业总收入(元)'])
train_num['注册资本（万元）']=np.log(train_num['注册资本（万元）'])
#%%
train_num.to_csv("data/train_values0422.csv",index=False,na_rep="NULL",encoding='utf_8_sig')
test_num.to_csv("data/test_values0422.csv",index=False,na_rep="NULL",encoding='utf_8_sig')