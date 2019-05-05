# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 09:57:38 2019

@author: 29132
"""

import numpy as np
import pandas as pd
from def_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
#%%
data=np.arange(1001,4001)
df_train= pd.DataFrame(data,columns=['企业编号'])
#%%
scores=load_data('赛题1数据集/企业评分.xlsx')
#%%
develop=load_data('赛题1数据集/上市信息财务信息-成长能力指标.xlsx')
num_columns=['企业编号','营业总收入(元)']
new_develop=develop[num_columns].copy()
new_develop=convert_invalid_value_to_null(new_develop)
new_develop=convert_str_to_same_unit(new_develop)
new_develop=new_develop.groupby('企业编号',as_index=False).sum()
#%%
operate=load_data('赛题1数据集/上市信息财务信息运营能力指标.xlsx')
operate=deal_with_finance_table(operate)
operate.drop('日期',axis=1,inplace=True)
sum_columns=['企业编号','存货周转天数(天)','总资产周转率(次)']
sum_operate=operate[sum_columns].copy()
operate_sum=sum_operate.groupby('企业编号',as_index=False).sum()
skew_columns=['应收账款周转天数(天)']
skew_operate=operate[skew_columns].copy()
skew_operate['企业编号']=operate['企业编号']
operate_skew=skew_operate.groupby('企业编号').skew().reset_index()
new_operate_index=pd.merge(operate_sum,operate_skew,on='企业编号',how='inner')
#%%
stock_index=load_data('赛题1数据集/上市公司财务信息-每股指标.xlsx')
stock_index=deal_with_finance_table(stock_index)
stock_index.drop(['日期'],axis=1,inplace=True)
new_stock_index=stock_index.groupby('企业编号',as_index=False).sum()
#%%
risk_index=load_data('赛题1数据集/上市信息财务信息-财务风险指标.xlsx')
risk_index=deal_with_finance_table(risk_index)
risk_index.drop(['日期','流动负债/总负债(%)'],axis=1,inplace=True)
risk_index['流动比率']=risk_index.groupby('企业编号')['流动比率'].transform('sum')
risk_index['资产负债率(%)']=risk_index.groupby('企业编号')['资产负债率(%)'].transform('std')
risk_index['速动比率']=risk_index.groupby('企业编号')['速动比率'].transform('skew')
new_risk_index=risk_index.drop_duplicates()
#%%
profitability_index=load_data('赛题1数据集/上市信息财务信息盈利能力指标.xlsx')
remain_columns=['企业编号','毛利率(%)','实际税率(%)']
profitability_index=profitability_index[remain_columns].copy()
for i in profitability_index.columns[1:]:
    profitability_index[i]=profitability_index[i].apply(convert_percent)
    profitability_index[i]=profitability_index[i].apply(pd.to_numeric, errors='coerce')
new_profitability_index=profitability_index.groupby('企业编号').skew().reset_index()
#%%
products=load_data('赛题1数据集/产品.xlsx')
new_products=products.groupby('企业编号',as_index=False).count()
new_products=pd.merge(df_train,new_products,on='企业编号',how='outer')
new_products.fillna(0,inplace=True)
#%%
arank_year=load_data('赛题1数据集/纳税A级年份.xlsx')
arank_year=arank_year.groupby('企业编号',as_index=False).count()
new_arank=pd.merge(df_train,arank_year,on='企业编号',how='outer')
new_arank.fillna(0,inplace=True)
#%%工商基本信息表
infor=load_data('赛题1数据集/工商基本信息表.xlsx')
infor['经营状态'].replace('存续（在营、开业、在册）','存续',inplace=True)
infor['经营状态'].replace('存续(在营、开业、在册)','存续',inplace=True)
infor['经营状态'].replace('在营（开业）企业','在业',inplace=True)
remain_columns=['企业编号','注册资本（万元）','行业大类（代码）','类型','经营状态']
new_infor=infor[remain_columns].copy()
#%%
land_mortgage=load_data('赛题1数据集/购地-市场交易-土地抵押.xlsx') 
num_columns= [f for f in land_mortgage.columns if land_mortgage.dtypes[f] != 'object'] 
new_land_mortgage=land_mortgage[num_columns].groupby('企业编号',as_index=False).sum()
new_land_mortgage=pd.merge(df_train,new_land_mortgage,on='企业编号',how='outer')
new_land_mortgage.fillna(0,inplace=True)
new_land_mortgage['是否购地']=new_land_mortgage['土地面积'].apply(lambda x: 'Yes' if x > 0 else 'No')
land_use_table=pd.read_excel('赛题1数据集/土地使用表.xlsx',dtype='object')
land_use_dict={}
for i in range(land_use_table.shape[0]):
    land_use_dict[land_use_table.iloc[i,0]]=land_use_table.iloc[i,1]
for i in range(land_mortgage.shape[0]):
    value=land_mortgage['土地用途'].iloc[i]
    if value in land_use_dict.keys():
        land_mortgage['土地用途'].replace(value,land_use_dict[value],inplace=True)       
land_mortgage['土地用途'].replace('05、07',land_use_dict['05'],inplace=True)
land_mortgage['土地用途'].replace('073',land_use_dict['07'],inplace=True)
land_mortgage['土地用途'].replace('074',land_use_dict['07'],inplace=True)
land_mortgage['土地用途'].replace('077',land_use_dict['07'],inplace=True)
str_columns=['企业编号','土地用途']
remain_land=land_mortgage[str_columns].copy()
#%%
land_transfer=load_data('赛题1数据集/购地-市场交易-土地转让.xlsx')
land_transfer_price=pd.concat([land_transfer['企业编号'],land_transfer['转让价格(万元)']],axis=1)
new_transfer_price=land_transfer_price.groupby('企业编号',as_index=False).sum()
#%%
credit_index = pd.read_excel('赛题1数据集/海关进出口信用.xlsx')
remain_columns=['企业编号','信用等级']
remain_credit=credit_index[remain_columns]
#%%
bond=load_data('赛题1数据集/债券信息.xlsx')
str_columns=['企业编号','债券品种','付息方式']
remain_bond=bond[str_columns].copy()
new_bond=remain_bond.groupby('企业编号',as_index=False).count()
new_bond=pd.merge(df_train,new_bond,on='企业编号',how='outer')
new_bond.fillna(0,inplace=True)
new_bond['是否有债券']=new_bond['债券品种'].apply(lambda x: 'Yes' if x > 0 else 'No')
new_bond.drop(['债券品种','付息方式'],axis=1,inplace=True)
#%%
compete_goods=load_data('赛题1数据集/竞品.xlsx')
compete_goods=deal_with_null(compete_goods)
compete_goods['竞品的标签']=compete_goods['竞品的标签'].str.split('\n',expand=True)[0]
A=compete_goods['竞品的标签'].value_counts().reset_index()
B=A[A['竞品的标签']>100]
B=B['index'].values.tolist()
compete_goods=compete_goods[compete_goods['竞品的标签'].isin(B)]
str_columns=['企业编号','竞品运营状态','竞品的标签']
remain_compete=compete_goods[str_columns].copy()
#%%
finance=load_data('赛题1数据集/融资信息.xlsx')
remain_columns=['企业编号','轮次']
remain_finance=finance[remain_columns].copy()
#%%
trademark=load_data('赛题1数据集/商标.xlsx')
trademark=delete_time(trademark)
new_trademark=trademark.groupby('企业编号',as_index=False).count()
new_trademark=pd.merge(df_train,new_trademark,on='企业编号',how='outer')
new_trademark.fillna(0,inplace=True)
new_trademark['是否有商标']=new_trademark['商标状态'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
bid=load_data('赛题1数据集/招投标.xlsx')
drop_columns=['公告类型','省份','发布时间']
new_bid=bid.drop(drop_columns,axis=1).groupby('企业编号',as_index=False).count()
new_bid=pd.merge(df_train,new_bid,on='企业编号',how='outer')
new_bid.fillna(0,inplace=True)
new_bid['是否有招投标']=new_bid['中标或招标'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
patents=load_data('赛题1数据集/专利.xlsx')
patents.drop(['授权公告日','申请日'],axis=1,inplace=True)
remain_patents=patents
new_patents=patents.groupby('企业编号',as_index=False).count()
new_patents=pd.merge(df_train,new_patents,on='企业编号',how='outer')
new_patents.fillna(0,inplace=True)
new_patents['是否有专利']=new_patents['专利类型'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
qc=load_data('赛题1数据集/资质认证.xlsx')
qc=delete_time(qc)
new_qc=qc.groupby('企业编号',as_index=False).count()
new_qc=pd.merge(df_train,new_qc,on='企业编号',how='outer')
new_qc.fillna(0,inplace=True)
new_qc['是否有资质']=new_qc['证书名称'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
copyrights=load_data('赛题1数据集/作品著作权.xlsx')
remain_columns=['企业编号','作品著作权类别']
new_copyrights=copyrights[remain_columns].groupby('企业编号',as_index=False).count()
new_copyrights=pd.merge(df_train,new_copyrights,on='企业编号',how='outer')
new_copyrights.fillna(0,inplace=True)
new_copyrights['是否有作品']=new_copyrights['作品著作权类别'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
software_copyrights = load_data('赛题1数据集/软著著作权.xlsx')
software_copyrights.drop(columns=['软件著作权版本号','软件著作权登记批准日期'],inplace=True)
new_software_copyrights=software_copyrights.groupby('企业编号',as_index=False).count()
new_software_copyrights=pd.merge(df_train,new_software_copyrights,on='企业编号',how='outer')
new_software_copyrights.fillna(0,inplace=True)
new_software_copyrights['是否有版权']=new_software_copyrights['软件全称'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
annual_web=load_data('赛题1数据集/年报-网站或网点信息.xlsx')
new_annual_web=annual_web.groupby('企业编号',as_index=False).count()
new_annual_web=pd.merge(df_train,new_annual_web,on='企业编号',how='outer')
new_annual_web.fillna(0,inplace=True)

#%%
annual_infor=load_data('赛题1数据集/年报-股东（发起人）及出资信息.xlsx')
new_annual_infor=annual_infor.groupby('企业编号',as_index=False).count()
new_annual_infor=pd.merge(df_train,new_annual_infor,on='企业编号',how='outer')
new_annual_infor.fillna(0,inplace=True)
new_annual_infor['是否认缴']=new_annual_infor['实缴出资信息'].apply(lambda x: 'Yes' if x > 0 else 'No')
#%%
train_num_features=[new_develop,new_operate_index,new_risk_index,\
                    new_stock_index,new_profitability_index,new_products,\
                    new_arank,new_infor,new_land_mortgage,new_copyrights,\
                    new_transfer_price,new_bond,new_trademark,new_bid,new_annual_infor,\
                    new_patents,new_qc,new_software_copyrights,new_annual_web]
#%%
data=np.arange(1001,4001)
train_num_data = pd.DataFrame(data,columns=['企业编号'])
for i in range(len(train_num_features)):
    train_num_data=pd.merge(train_num_data,train_num_features[i],on='企业编号',how='outer')
train_score=pd.merge(train_num_data,scores,on='企业编号',how='inner')
#%%
train_score.to_csv("data/train_num0422.csv",index=False,na_rep="NULL",encoding='utf_8_sig')
#%%
remain_bond.to_csv('data/train_remain_bond0422.csv',index=False,na_rep="NULL",encoding='utf_8_sig')
remain_compete.to_csv('data/train_remain_compete0422.csv',index=False,na_rep="NULL",encoding='utf_8_sig')
remain_credit.to_csv('data/train_remain_credit0422.csv',index=False,na_rep="NULL",encoding='utf_8_sig')
remain_land.to_csv('data/train_remain_land0422.csv',index=False,na_rep="NULL",encoding='utf_8_sig')
remain_finance.to_csv('data/train_remain_finance0422.csv',index=False,na_rep="NULL",encoding='utf_8_sig')

