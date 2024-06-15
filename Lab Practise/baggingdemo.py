import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import os 
import pandas as pd
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split , GridSearchCV , KFold, RandomizedSearchCV
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import  DecisionTreeClassifier,RandomForestClassifier
from sklearn.ensemble import  BaggingClassifier
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,accuracy_score

kyp = pd.read_csv(r"C:\Users\dai\Desktop\Machine Learning\Cases\Cases\Kyphosis\Kyphosis.csv")

# kyp_ind = list(kyp.index)

# samp_ind = np.random.choice(kyp_ind , size=60 , replace=False)

# samp_kyp = kyp.iloc[samp_ind,:]


# samp_ind = np.random.choice(kyp_ind , size=81 , replace=True)

# samp_kyp = kyp.iloc[samp_ind,:]
# samp_kyp


le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)

lr = LogisticRegression()
bagg = BaggingClassifier(lr, n_estimators=25, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24, stratify = y)

bagg.fit(X_train,y_train)
y_pred = bagg.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba =  bagg.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))
      
dtr = DecisionTreeClassifier(random_state=24)
bagg = BaggingClassifier(dtr, n_estimators=25, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24, stratify = y)

bagg.fit(X_train,y_train)
y_pred = bagg.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba =  bagg.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))
      

dtr = RandomForestClassifier(random_state=24)
bagg = BaggingClassifier(dtr, n_estimators=25, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24, stratify = y)

bagg.fit(X_train,y_train)
y_pred = bagg.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba =  bagg.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))
      
    
                                          