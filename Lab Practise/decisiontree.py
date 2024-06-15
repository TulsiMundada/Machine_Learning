import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV ,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
kyp = pd.read_csv("C:/Users\dai\Desktop\Machine Learning/Cases/Cases/Kyphosis/Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)
dtc = DecisionTreeClassifier(random_state=24)
dtc.fit(X, y)

plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show() 

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))

#################
params={'min_samples_split':[4,6,10,20],
        'min_samples_leaf':[1,5,10,15],
        'max_depth':[None,4,3,2]
        }

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
              random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, cv =kfold ,scoring = 'neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


best_tree = gcv.best_estimator_
plt.figure(figsize=(25,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show() 




























