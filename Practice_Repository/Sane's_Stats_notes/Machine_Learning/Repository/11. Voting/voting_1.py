import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import numpy as np 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

svm_l = SVC(kernel="linear", 
          probability=True, random_state=24)
std_scaler = StandardScaler()
pipe_l = Pipeline([('SCL', std_scaler),('SVM',svm_l)])

svm_r = SVC(kernel="rbf", 
          probability=True, random_state=24)
std_scaler = StandardScaler()
pipe_r = Pipeline([('SCL', std_scaler),('SVM',svm_r)])

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

voting = VotingClassifier([('LR',lr),('SVML',pipe_l),
                           ('SVM_R', pipe_r),('LDA',lda),
                           ('TREE', dtc)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify=y)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))

#########################################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
print(voting.get_params())

params = {'SVML__SVM__C': np.linspace(0.001, 3, 5),
          'SVM_R__SVM__C':np.linspace(0.001, 3, 5),
          'SVM_R__SVM__gamma':np.linspace(0.001, 3, 5),
          'LR__C': np.linspace(0.001, 3, 5),   
          'TREE__max_depth':[None, 3, 2] }
gcv = GridSearchCV(voting, param_grid=params,
                   cv=kfold, scoring='neg_log_loss',
                   n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)




