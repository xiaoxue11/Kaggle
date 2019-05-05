# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:14:44 2019

@author: 29132
"""
import pandas as pd


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

def deal_with_name_attribute(df):
    df['Name']=df['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
    df['Name']=df['Name'].apply(change_name)
    df.loc[(df['Name']=='Dr')&(df['Sex']=='female'),'Name']='Mrs'
    df.loc[(df['Name']=='Dr')&(df['Sex']=='male'),'Name']='Mr'
    return df  

def ageclass_estimator(value):
    if (value<=10.0):
        new_value='child'
    elif (value>10.0) &(value<=30.0):
        new_value='youth'
    elif (value>30.0) &(value<=60.0):
        new_value='adult'
    elif value>60.0:
        new_value='senior'
    return new_value

def deal_with_missing_age(df):
    name_age=df.groupby('Name')['Age'].mean().reset_index()
    df['Age_fillna']=df['Age']
    df.loc[(df['Age_fillna'].isnull())&(df['Name']=='Master'),'Age_fillna']=name_age['Age'][0]
    df.loc[(df['Age_fillna'].isnull())&(df['Name']=='Miss'),'Age_fillna']=name_age['Age'][1]
    df.loc[(df['Age_fillna'].isnull())&(df['Name']=='Mr'),'Age_fillna']=name_age['Age'][2]
    df.loc[(df['Age_fillna'].isnull())&(df['Name']=='Mrs'),'Age_fillna']=name_age['Age'][3]
    df['Age_fillna']=df['Age_fillna'].apply(ageclass_estimator)
    return df

def deal_with_cabin_missing(df):
    df['Cabin'].fillna('Z',inplace=True)
    df['Cabin_mark']=df['Cabin'].apply(lambda x:'yes'if(x=='Z') else 'No')
    df['Cabin']=df['Cabin'].apply(lambda x:x[0])
    return df

def deal_with_missing_fare(df):
    pclass_fare=df.groupby('Pclass')['Fare'].median().reset_index()
    df.loc[(df['Fare'].isnull())&(df['Pclass']==1),'Fare']=pclass_fare['Fare'][0]
    df.loc[(df['Fare'].isnull())&(df['Pclass']==2),'Fare']=pclass_fare['Fare'][1]
    df.loc[(df['Fare'].isnull())&(df['Pclass']==3),'Fare']=pclass_fare['Fare'][2]
    return df

def Fare_class(value):
    if (value<=8.0):
        new_value='0'
    elif (value>8.0) &(value<=32.0):
        new_value='1'
    elif (value>32.0) &(value<=100.0):
        new_value='2'
    elif value>100.0:
        new_value='3'
    return new_value

def person_level(value):
    if value<=8.0:
        new_value='low'
    elif (value>8.0) &(value<=30.0):
        new_value='median'
    elif value>30.0:
        new_value='high'
    return new_value

def clean_transform_data(df):
    df=deal_with_name_attribute(df)
    df=deal_with_missing_age(df)
    df=deal_with_cabin_missing(df)
    df=deal_with_missing_fare(df)
    df['Fare_class']=df['Fare'].apply(Fare_class)
    df['family_people']=df['SibSp']+df['Parch']+1
    df['Fare_per_people']=df['Fare']/df['family_people']
    df['person_alone'] = df['family_people'].apply(lambda x:'No' if x>1 else 'Yes') 
    df['Pclass']=df['Pclass'].apply(str)
    df['Fare_per_people']=df['Fare_per_people'].apply(person_level)
    return df

