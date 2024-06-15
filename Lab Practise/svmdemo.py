import pandas as pd
import numpy as np
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score 


kyp = pd.read_csv("Cases/Cases/Kyphosis/Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)

svm= SVC(C=0.1, kernel="linear")

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)

svm.fit(X_train, y_train)
y_pred= svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

##############
params={'C':[0.1, 1, 0.5, 2, 3]}
kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)


gcv = GridSearchCV(svm, param_grid=params, cv=kfold )
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
##########################
poly = PolynomialFeatures(degree=1).set_output(transform="pandas")
lr = LinearRegression()
pipe = Pipeline([("POLY",poly),('LR',lr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))



























