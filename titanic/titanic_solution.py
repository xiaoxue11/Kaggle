# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:40:19 2019

@author: 29132
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('./input/train.csv')
test=pd.read_csv('./input/test.csv')
#%%
def change_name(value):
    new_name='unknown'
    if value in [' Mr',' Don',' Rev',' Major',' Sir',' Jonkheer',' Col',' Capt']:
        new_name='Mr'
    elif value in [' Mrs',' Mme',' the Countess',' Lady',' Dona']:
        new_name='Mrs'
    elif value in [' Miss',' Ms',' Mlle']:
        new_name='Miss'
    elif value==' Master':
        new_name='Master'
    elif value==' Dr':
        new_name='Dr'
    return new_name
#%%
train['Name']=train['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
train['Name']=train['Name'].apply(change_name)
#%%
test['Name']=test['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
#%%
test['Name']=test['Name'].apply(change_name)
#%%
values1=np.average(train[train['Name']=='Master']['Age'].dropna())
values2=np.average(train[train['Name']=='Miss']['Age'].dropna())
values3=np.average(train[train['Name']=='Mr']['Age'].dropna())
values4=np.average(train[train['Name']=='Mrs']['Age'].dropna())
values5=np.average(train[train['Name']=='Dr']['Age'].dropna())
#%%
values=train.groupby('Name')['Age'].mean().reset_index()
#%%
train.loc[(train['Age'].isnull())&(train['Name']=='Dr'),'Age']=values['Age'][0]
train.loc[(train['Age'].isnull())&(train['Name']=='Master'),'Age']=values['Age'][1]
train.loc[(train['Age'].isnull())&(train['Name']=='Miss'),'Age']=values['Age'][2]
train.loc[(train['Age'].isnull())&(train['Name']=='Mr'),'Age']=values['Age'][3]
train.loc[(train['Age'].isnull())&(train['Name']=='Mrs'),'Age']=values['Age'][4]
#%%
train.loc[train['Cabin'].notnull(),'Cabin']=1.0
train.loc[train['Cabin'].isnull(),'Cabin']=0.5
#%%
value1=train.groupby('Pclass')['Fare'].median().reset_index()
train.loc[(train['Fare'].isnull())&(train['Pclass']==1),'Fare']=value1['Fare'][0]
train.loc[(train['Fare'].isnull())&(train['Pclass']==2),'Fare']=value1['Fare'][1]
train.loc[(train['Fare'].isnull())&(train['Pclass']==3),'Fare']=value1['Fare'][2]
#%%
test.loc[(test['Fare'].isnull())&(test['Pclass']==1),'Fare']=value1['Fare'][0]
test.loc[(test['Fare'].isnull())&(test['Pclass']==2),'Fare']=value1['Fare'][1]
test.loc[(test['Fare'].isnull())&(test['Pclass']==3),'Fare']=value1['Fare'][2]
#%%
train['family_people']=train['SibSp']+train['Parch']+1
test['family_people']=test['SibSp']+train['Parch']+1
train['Fare_per_people']=train['Fare']/train['family_people']
test['Fare_per_people']=test['Fare']/train['family_people']
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)