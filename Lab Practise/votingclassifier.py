import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV ,KFold
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

kyp = pd.read_csv("C:/Users\dai\Desktop\Machine Learning/Cases/Cases/Kyphosis/Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)

svm_1 = SVC(kernel="linear",
            probability = True , random_state=24
            )
std_scaler = StandardScaler()
pipe_1 = Pipeline([('SCL',std_scaler),('SVM',svm_1)])

svm_r = SVC(kernel="rbf",
            probability = True , random_state=24
            )
std_scaler = StandardScaler()
pipe_r = Pipeline([('SCL',std_scaler),('SVM',svm_r)])

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

voting = VotingClassifier([('LR',lr),('SVML',pipe_1),('SVMR',pipe_r),
                           ('LDA',lda),('TREE',dtc)],voting='soft')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)

voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_proba = voting.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_proba))
############
kfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state=24)
#print(voting.get_params())

params={'SVML__SVM__C':np.linspace(0.001, 3,5),
        'TREE__max_depth':[None,3,2],
        'SVMR__SVM__C':np.linspace(0.001, 3,5),
        'SVMR__SVM__gamma':np.linspace(0.001, 3,5),
        'LR__C':np.linspace(0.001,3,5),
        }


gcv = GridSearchCV(voting, param_grid=params, cv=kfold ,scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
##############

# params={'SVML__SVM__C':np.linspace(0.001, 3,10),
#         'Tree_max_depth':[None,3,2],
        # 'SVML__SVM__gamma':np.linspace(0.001, 3,10)
#     }























