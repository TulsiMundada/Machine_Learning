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
glass = pd.read_csv("Cases/Cases/Glass Identification/Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop(['Type'],axis=1)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify = y)
dtc = DecisionTreeClassifier(random_state=24,min_samples_split=4)

params={'min_samples_split':[2,35,5],
        'min_samples_leaf':[1,10,15],
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
               class_names=list(le.classes_),
               filled=True,fontsize=18)
plt.show() 
#################
df_imp = pd.DataFrame({'Feature':list(X.columns),'Importance':best_tree.feature_importances_})
plt.bar(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importance")
plt.show()

m_left , m_right = 183 , 31
g_ledt , g_right= 0.679 , 0.287
m = 214

ba_split = (m_left/m)*g_ledt + (m_right/m)*g_right



#######HR ANALYSIS

hr = pd.read_csv(r"C:/Users\dai\Desktop\Machine Learning\Cases\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr,drop_first=True)
X = dum_hr.drop('left',axis=1)
y = dum_hr['left']

params={'min_samples_split':[2,35,5],
        'min_samples_leaf':[1,35,5],
        'max_depth':[None,4,3,2]
        }

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
              random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, cv =kfold ,scoring = 'neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(25,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=18)
plt.show() 

df_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':best_tree.feature_importances_})
plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importance")
plt.show()




















