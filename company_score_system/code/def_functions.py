# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:53:43 2019

@author: 29132
"""
import re
import numpy as np
import pandas as pd
import difflib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def load_data(filename):
    df = pd.read_excel(filename,set='\n',enconding='utf8')
    df.duplicated()
    df=df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df
def find_str_columns(df):
    str_columns=[f for f in df.columns if df.dtypes[f] == 'object']
    return str_columns

def find_num_columns(df):
    num_columns=[f for f in df.columns if df.dtypes[f] != 'object'] 
    return num_columns

def columns_classify(df):
    str_columns = [f for f in df.columns if df.dtypes[f] != 'object']
    num_columns = [f for f in df.columns if df.dtypes[f] == 'object']
    return num_columns,str_columns

def convert_numeric_to_string(value):
    return str(value)

def convert_invalid_value_to_null(df):
    df.replace('--', np.nan,inplace=True)
    df.replace('--%', np.nan,inplace=True)
    df.replace('-', np.nan,inplace=True)
    df.replace('0000000001',np.nan,inplace=True)
    df.replace('企业选择不公示',np.nan,inplace=True)
    df.replace('选择不公示',np.nan,inplace=True)
    df.replace('万元',np.nan,inplace=True)
    df.replace('人',np.nan,inplace=True)
    df.replace('未披露',np.nan,inplace=True)
    df.replace('期间',np.nan,inplace=True)
    df.replace('期限',np.nan,inplace=True)
    df.replace('限期',np.nan,inplace=True)
    df.replace('未约定',np.nan,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : 'percent of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[\
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('percent of Total Values', ascending=False)
        return mis_val_table_ren_columns
    
def convert_object_to_value(df):
    le = LabelEncoder()
    le_count = 0
    for col in df:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1           
    return df

def deal_with_null(df):
    mis_val_table=missing_values_table(df)
    mis_val_table=mis_val_table[mis_val_table['percent of Total Values']>0.75]
    df_new=df.drop(mis_val_table.index.tolist(),axis=1)
    return df_new

def delete_one_attribute(df):
    delete_columns=[]
    for i in range(df.shape[1]):
        value_num=len(df.iloc[:,i].value_counts())
        if value_num==1:
            delete_columns.append(df.columns[i])
    df.drop(columns=delete_columns,inplace=True)
    return df

def convert_time_to_year(df):
    str_pattern='时间|日期'
    pattern = re.compile(str_pattern)
    for i in df.columns:
        m=pattern.search(i)
        if m:
            df['year']=pd.to_datetime(df[i],yearfirst=True).dt.year.apply(convert_numeric_to_string)
    return df

def calculate_last_days(df):
    str_pattern='时间|日期'
    pattern = re.compile(str_pattern)
    now = datetime.now()
    for i in df.columns:
        m=pattern.search(i)
        if m:
            df['last_days']=(now-pd.to_datetime(df[i],yearfirst=True)).dt.days
    return df

def delete_time(df):
    str_pattern='时间|日期'
    time_columns=[]
    pattern = re.compile(str_pattern)
    for i in range(len(df.columns)):
        str_to_match=df.columns[i]
        m=pattern.search(str_to_match)
        if m:
            time_columns.append(str_to_match)
    df.drop(columns=time_columns,inplace=True)
    return df

def convert_str_to_same_unit(df):
    str_columns=find_str_columns(df)
    remain_df=df.drop(str_columns,axis=1)
    deal_df=df[str_columns].copy()
    zhpattern = re.compile(u'[\u4e00-\u9fa5]+')
    for i in str_columns:
        for j in deal_df.index:
            if (deal_df[i].notnull())[j]:
                str_to_match=deal_df[i][j]
                m1=zhpattern.search(str_to_match)
                if not m1:
                    new_value=re.sub("[%&',;=?$\x22]+",'',str_to_match)
                    deal_df[i][j]=new_value
                else:
                    remian_value=re.sub(u'[\u4E00-\u9FA5]', "", str_to_match)
                    new_value=re.sub("[%&',;=?$\x22]+",'',remian_value)
                    if str_to_match[m1.start(): m1.end()]=='万':
                        new_value=float(new_value)*10**(-4)
                    elif str_to_match[m1.start(): m1.end()]=='万亿': 
                        new_value=float(new_value)*10**3
                    deal_df[i][j]=new_value
    deal_df=deal_df.apply(pd.to_numeric, errors='ignore')
    df_new=pd.merge(remain_df, deal_df, left_index=True, right_index=True, how='inner')
    return df_new

def deal_with_missing_value(df):
    null_infor=df.isnull().sum().sort_values(ascending=False)
    null_columns=null_infor[null_infor.values>0].index.tolist()
    str_columns=find_str_columns(df)
    num_columns=find_num_columns(df)
    if not num_columns:
        for i in null_columns:
            pad_value=df[i].value_counts().idxmax()
            df[i].fillna(pad_value,inplace=True)
        return df
    else:
        full_num_columns=list(set(num_columns).difference(set(null_columns)))
        deal_num_columns=list(set(null_columns).intersection(set(num_columns)))
        deal_str_columns=list(set(null_columns).intersection(set(str_columns)))
        if  not full_num_columns:
            pass
        else:
            if deal_num_columns:
                for i in deal_num_columns:
                    process_df=df[full_num_columns].copy()
                    process_df[i]=df[i].copy()
                    known_value=process_df[df[i].notnull()].values
                    unknown_value = process_df[df[i].isnull()].values
                    y = known_value[:, -1]
                    X = known_value[:, :-1]
                    rfr=RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
                    rfr.fit(X, y)
                    predicted_value = rfr.predict(unknown_value[:,:-1])
                    df.loc[(df[i].isnull()), i] = predicted_value
            if deal_str_columns:
                for i in deal_str_columns:
                    str_df=df[full_num_columns]
                    str_df[i]=df[i].copy()
                    known_value=str_df[df[i].notnull()].values
                    unknown_value = str_df[df[i].isnull()].values
                    y = known_value[:, -1]
                    X = known_value[:, :-1]
                    clf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0)
                    clf.fit(X, y)
                    predicted_value = clf.predict(unknown_value[:,:-1])
                    df.loc[(df[i].isnull()), i] = predicted_value
    return df

def remove_similar_columns(df):
    similar_columns=[]
    str_columns=find_str_columns(df)
    for i in range(len(str_columns)-1):
        for j in range(i+1,len(str_columns)):
            str1=df[str_columns[i]].values
            str2=df[str_columns[j]].values
            seq = difflib.SequenceMatcher(None, str1, str2)
            ratio = seq.ratio()
            if ratio>0.9:
                similar_columns.extend([str_columns[i],str_columns[j]])
    df.drop(similar_columns[1:],axis=1,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
def deal_with_finance_table(df):
    df=delete_one_attribute(df)
    df=convert_invalid_value_to_null(df)
    df=df.apply(pd.to_numeric, errors='ignore')
    columns=['企业编号','日期']
    remain_df=df[columns].copy()
    deal_df=df.drop(columns,axis=1)
    deal_df=convert_str_to_same_unit(deal_df)
    deal_df.dropna(how='all',inplace=True)
    deal_df=deal_with_missing_value(deal_df)
    df_new=pd.merge(remain_df,deal_df,left_index=True, right_index=True, how='inner')
    return df_new
def convert_percent(value):
    new_value=value.replace('%','')
    return new_value

def plot_feature_importances(df, threshold = 0.95): 
    plt.rcParams['font.size'] = 18   
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df

def convert_values_to_one(value):
    if value>0:
        value=value/value
    return int(value)