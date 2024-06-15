import pandas as pd
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
glass = pd.read_csv("Cases/Cases/Glass Identification/Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop(['Type'],axis=1)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis(reg_param=0.5)
lda.fit(X_train,y_train)
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))

qda.fit(X_train,y_train)
y_pred = qda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))

kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
result= cross_val_score(lda,X,y,cv=kfold,scoring='neg_log_loss')
print(result.mean())


##############
brupt = pd.read_csv("Cases/Cases/Bankruptcy/Bankruptcy.csv")
le = LabelEncoder()
y = brupt['D']
X = brupt.drop(['NO','D'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
lda.fit(X_train,y_train)
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))

qda.fit(X_train,y_train)
y_pred = qda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))

kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
result= cross_val_score(lda,X,y,cv=kfold,scoring='neg_log_loss')
print(result.mean())


kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
result= cross_val_score(qda,X,y,cv=kfold,scoring='neg_log_loss')
print(result.mean())







