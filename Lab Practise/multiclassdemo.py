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

train = pd.read_csv(r"C:\Users\dai\Desktop\Machine Learning\multiclass\train.csv",index_col=0)
test = pd.read_csv(r"C:\Users\dai\Desktop\Machine Learning\multiclass\test.csv")
dum_tr = pd.get_dummies(train,drop_first=True)
X = dum_tr.drop('Status',axis=1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform()

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
               class_names=['0','1','2'],
               filled=True,fontsize=18)
plt.show() 

df_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':best_tree.feature_importances_})
plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importance")
plt.show()