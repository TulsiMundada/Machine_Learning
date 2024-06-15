import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split , KFold 
from sklearn.metrics import r2_score 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  cross_val_score
##### Boston
boston = pd.read_csv("Datasets/Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)
#X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)
poly = PolynomialFeatures(degree=1).set_output(transform="pandas")
lr = LinearRegression()
pipe = Pipeline([("POLY",poly),('LR',lr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True,  random_state=24)
degrees = [1,2,3,4,5]
scores = []
lr = LinearRegression()
for i in degrees:
    poly = PolynomialFeatures(degree=i) 
    pipe = Pipeline([("POLY",poly),('LR',lr)])
    results = cross_val_score(pipe, X,y,cv=kfold)
    scores.append(results.mean())
    
i_max = np.argmax(scores)
print("Best Degree = ", degrees[i_max])    
print("Best Score = ", scores[i_max])    
    
    
    
    
    
    
    
    
    
    