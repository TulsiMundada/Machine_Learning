import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np 

df = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")

y = df['Strength']
X = df.drop('Strength', axis=1)

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVR(kernel="linear")
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 3) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__epsilon':np.linspace(0.001,5,3) }

kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv_lin = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)
gcv_lin.fit(X, y)
pd_cv = pd.DataFrame( gcv_lin.cv_results_ )
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

#### Poly
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="poly", probability=True, random_state=24)
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__degree':[2,3],
          'SVM__coef0':np.linspace(0, 3, 5),
          'SVM__decision_function_shape':['ovo','ovr']}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv_poly = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=2)
gcv_poly.fit(X, y)
pd_cv = pd.DataFrame( gcv_poly.cv_results_ )
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)

#### Radial
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="rbf", probability=True, random_state=24)
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__gamma':np.linspace(0.001, 5, 5),
          'SVM__decision_function_shape':['ovo','ovr']}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv_rbf = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=2)
gcv_rbf.fit(X, y)
pd_cv = pd.DataFrame( gcv_rbf.cv_results_ )
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)

print("Linear Kernel Results:")
print("Params:",gcv_lin.best_params_)
print("Score:",gcv_lin.best_score_)

print("Polynomial Kernel Results:")
print("Params:",gcv_poly.best_params_)
print("Score:",gcv_poly.best_score_)

print("Radial Kernel Results:")
print("Params:",gcv_rbf.best_params_)
print("Score:",gcv_rbf.best_score_)





