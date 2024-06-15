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


sat = pd.read_csv("C:/Users\dai\Desktop\Machine Learning\Cases\Cases\Satellite Imaging\Satellite.csv", sep=';')
print(sat.head())

le = LabelEncoder()
y = sat['classes']
X = sat.drop(['classes'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train,y_train)

y_pred = lda.predict(X_test)
print("accuracy score",accuracy_score(y_test, y_pred))
y_pred_proba = lda.predict_proba(X_test)
print("log loss",log_loss(y_test, y_pred_proba))

qda.fit(X_train,y_train)

y_pred = qda.predict(X_test)
print("accuracy score",accuracy_score(y_test, y_pred))
y_pred_proba = qda.predict_proba(X_test)
print("log loss",log_loss(y_test, y_pred_proba))

kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
result= cross_val_score(lda,X,y,cv=kfold,scoring='neg_log_loss')
print("lda ",result.mean())


kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
result= cross_val_score(qda,X,y,cv=kfold,scoring='neg_log_loss')
print("qda",result.mean())







