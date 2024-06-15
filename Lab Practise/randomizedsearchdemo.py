import pandas as pd
import numpy as np
from sklearn.metrics import  r2_score
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split , GridSearchCV , KFold, RandomizedSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.linear_model import Ridge , Lasso
from sklearn.ensemble import  VotingRegressor


conc = pd.read_csv("C:/Users\dai\Desktop\Machine Learning\Cases\Cases\Concrete Strength\Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']
lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()
dtr = DecisionTreeRegressor(random_state=24)
voting = VotingRegressor([('LR',lr),('RIDGE',ridge),('DTR',dtr),
                           ('LASSO',lasso)])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)

lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)

voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("LR :" , r2_lr)
print("Ridge :" , r2_ridge)
print("Lasso :" , r2_lasso)
print("Tree :" , r2_dtr)
print("Voting :" , r2_voting)

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),('DTR',dtr),
                           ('LASSO',lasso)])


kfold = KFold(n_splits = 5,shuffle=True,random_state=24)
#print(voting.get_params())

params={'DTR__min_samples_split':[2,5,10],
        'DTR__max_depth':[None,3,4,5],
        'DTR__min_samples_leaf':[1, 5 ,10],
        'RIDGE__alpha':np.linspace(0.001, 3,5),
        'LASSO__alpha':np.linspace(0.001,3,5),
        }


gcv = GridSearchCV(voting, param_grid=params, cv=kfold ,n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

kfold = KFold(n_splits = 5,shuffle=True,random_state=24)
#print(voting.get_params())
voting = VotingRegressor([('LR',lr),('RIDGE',ridge),('DTR',dtr),
                           ('LASSO',lasso)])

params={'DTR__min_samples_split':[2,4,5,8,10],
        'DTR__max_depth':[None,3,4,5],
        'DTR__min_samples_leaf':[1,4,5,8,10],
        'RIDGE__alpha':np.linspace(0.001, 3,5),
        'LASSO__alpha':np.linspace(0.001,3,5),
        }

rgcv = RandomizedSearchCV(voting, param_distributions=params, cv=kfold
                          ,n_jobs=-1 , n_iter =20 ,scoring='r2')
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)

best_model = rgcv.best_estimator_

