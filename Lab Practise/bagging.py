import pandas as pd
import numpy as np
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import os 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold


os.chdir(r"C:\Users\dai\Desktop\Machine Learning\Competitions\flood")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_lr.csv", index=False)
###############################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

el = ElasticNet()
bagg = BaggingRegressor(el, n_estimators=25, random_state=24)

print(bagg.get_params())

params = {'estimator__alpha':np.linspace(0.001, 5, 5), 
          'estimator__l1_ratio':np.linspace(0, 1, 4)}

gcv = GridSearchCV(bagg, param_grid=params, cv=kfold,verbose=3)
gcv.fit(X_train, y_train)



best_model = gcv.best_estimator_
y_pred = best_model.predict(X_test)

submit = pd.DataFrame({'id':test['id'], 'FloodProbabilty':y_pred})

submit.to_csv("sbt_bagg.csv", index=False)
