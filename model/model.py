from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import pandas as pd

models = [
    ('LogisticRegression', LogisticRegression()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('XGBClassifier', XGBClassifier()),
    #('GBM', GradientBoostingClassifier()),
    #('HGBM', HistGradientBoostingClassifier()),
]


def metric(y_true, y_pred, y_proba):
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall =  metrics.recall_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_proba[:,1])
        print(f' Accuracy : {accuracy:.2f} \n f1 : {f1:.2f} \n precision : {precision:.2f} \n recall : {recall:.2f} \n AUC : {auc:.2f} ')



def fit(models, X_train, y_train, X_test, y_test):
    proba = {}
    for model in models:
        print(model[0])
        model[1].fit(X_train, y_train)
        y_pred = model[1].predict(X_test)
        y_proba = model[1].predict_proba(X_test)
        proba[model[0]] = y_proba[:,1]
        metric(y_true = y_test, y_pred = y_pred, y_proba = y_proba)
        print("    -      -    -")
        print("    -      -    -")
    
    return proba


def optimise(models, X_train, y_train, score, cv):
    for model in models:
        print(model[0])
        param_grid = model[2]
        GridSearchCv(model, X_train, y_train,
                     cv = cv, scoring = score,
                     param_grid = param_grid)
        

