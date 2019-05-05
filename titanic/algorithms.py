# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:15:56 2019

@author: 29132
"""
import pandas as pd
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#%%
train_raw=pd.read_csv('./input/train.csv')
test_raw=pd.read_csv('./input/test.csv')
#%%
train=clean_transform_data(train_raw)
test=clean_transform_data(test_raw)
#%%
y=train['Survived']
train.drop(['PassengerId','Age','Ticket','Fare','Survived'],axis=1,inplace=True)
test_PassengerID=test['PassengerId']
test.drop(['PassengerId','Age','Ticket','Fare'],axis=1,inplace=True)
features = pd.concat([train, test]).reset_index(drop=True)
final_features = pd.get_dummies(features).reset_index(drop=True)
X = final_features.iloc[:len(y), :]
X_sub = final_features.iloc[len(y):, :]
#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
#%%
def cv_accuracy_scores(model,X_train,y_train):
    scores=cross_val_score(model,X_train, y_train, cv=kfolds, scoring="accuracy")
    return scores.mean()
def test_accuy(model,X_train,y_train,y_test):
    model.fit(X_train,y_train)
    y_test_pred=model.predict(X_test)
    scores=np.round(accuracy_score(y_test_pred,y_test),3)
    return scores
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
logreg_model=LogisticRegression(C=0.3, penalty='l2', tol=1e-6)
cv_LR_accuy=cv_accuracy_scores(logreg_model,X_train,y_train)
test_LR_accuy=test_accuy(logreg_model,X_train,y_train,y_test)

#%%
knn = KNeighborsClassifier(metric='minkowski', p=2)
cv_knn_accuy=cv_accuracy_scores(knn,X_train,y_train)
test_knn_accuy=test_accuy(knn,X_train,y_train,y_test)

#%%
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
cv_GB_accuy=cv_accuracy_scores(gaussian,X_train,y_train)
test_GB_accuy=test_accuy(gaussian,X_train,y_train,y_test)
#%%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] 
gammas = [0.0001,0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
#%%
SVM=SVC(kernel = 'rbf', probability=True,C=4,gamma=0.01)
cv_SVM_accuy=cv_accuracy_scores(SVM,X_train,y_train)
test_SVM_accuy=test_accuy(SVM,X_train,y_train,y_test)
#%%
from sklearn.tree import DecisionTreeClassifier
max_depth = range(1,30)
max_feature = [21,22,23,24,25,26,28,29,30,'auto']
criterion=["entropy", "gini"]
param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid = param, scoring='accuracy', \
                    cv=kfolds, verbose=1, n_jobs=4)
grid.fit(X_train, y_train) 
print(grid.best_score_)
print(grid.best_params_)
#%%
DTC=DecisionTreeClassifier(criterion='entropy',max_depth=4,max_features=28)
cv_DTC_accuy=cv_accuracy_scores(DTC,X_train,y_train)
test_DTC_accuy=test_accuy(DTC,X_train,y_train,y_test)
#%%
from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150,155,160];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
parameters = {'n_estimators':n_estimators,'max_depth':max_depth,'criterion': criterions}
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,scoring='accuracy', \
                                 cv=kfolds, verbose=1, n_jobs=4)
grid.fit(X_train,y_train) 
print(grid.best_score_)
print(grid.best_params_)
#%%
RFC=RandomForestClassifier(max_features='auto',n_estimators=155,max_depth=5,criterion ='gini')
cv_RFC_accuy=cv_accuracy_scores(RFC,X_train,y_train)
test_RFC_accuy=test_accuy(RFC,X_train,y_train,y_test)
#%%
from sklearn.ensemble import BaggingClassifier
n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];
parameters = {'n_estimators':n_estimators}
grid = GridSearchCV(BaggingClassifier(base_estimator= None,bootstrap_features=False),
                                 param_grid=parameters,scoring='accuracy', \
                                cv=kfolds, verbose=1, n_jobs=4)
grid.fit(X_train,y_train) 
print(grid.best_score_)
print(grid.best_params_)
#%%
BagClasifier=BaggingClassifier(base_estimator= None,bootstrap_features=False,n_estimators=150)
cv_BC_accuy=cv_accuracy_scores(BagClasifier,X_train,y_train)
test_BC_accuy=test_accuy(BagClasifier,X_train,y_train,y_test)
#%%
from sklearn.ensemble import AdaBoostClassifier
n_estimators = [100,140,145,150,160, 170,175,180,185];
learning_r = [0.1,1,0.01,0.5]
parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r}
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None),param_grid=parameters,\
                                 cv=kfolds, verbose=1, n_jobs=4)
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
#%%
ABC=AdaBoostClassifier(base_estimator= None,n_estimators=150,learning_rate=1)
cv_ABC_accuy=cv_accuracy_scores(ABC,X_train,y_train)
test_ABC_accuy=test_accuy(ABC,X_train,y_train,y_test)
#%%
from sklearn.ensemble import GradientBoostingClassifier
gradient_boost = GradientBoostingClassifier()
cv_gb_accuy=cv_accuracy_scores(gradient_boost,X_train,y_train)
test_gb_accuy=test_accuy(gradient_boost,X_train,y_train,y_test)
#%%
from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
cv_XGB_accuy=cv_accuracy_scores(XGBClassifier,X_train,y_train)
test_XGB_accuy=test_accuy(XGBClassifier,X_train,y_train,y_test)
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
confusion_matrix=confusion_matrix(y_train, y_train_pred)
precison=precision_score(y_train, y_train_pred)
recall=recall_score(y_train, y_train_pred)
f1=f1_score(y_train,y_train_pred)
#%%             
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

y_scores=cross_val_predict(logreg_model,X_train, y_train, cv=kfolds,method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper right")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

#%%
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
roc_auc_score(y_train, y_scores)
#%%
from sklearn.metrics import accuracy_score
def plot_learning_curves(model, X, y):
    X_train1, X_val, y_train1, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(2, len(X_train1)):
        model.fit(X_train1[:m], y_train1[:m])
        y_train_predict = model.predict(X_train1[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(1-accuracy_score(y_train[:m],y_train_predict))
        val_errors.append(1-accuracy_score(y_val,y_val_predict))
    plt.plot(train_errors, "r-+", linewidth=2, label="train")
    plt.plot(val_errors, "b-", linewidth=3, label="val")
    plt.legend(loc='best')
plot_learning_curves(logreg_model,X_train,y_train)