
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from dask.array.tests.test_array_core import test_size
from bokeh.core.properties import Align
from scipy.stats.mstats_basic import threshold
df_wine=pd.read_csv('~/odoo-dev/my_modules/BackUp/Machine_Learning/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid',
                 'Ash','Alcalinity of ash','Magnesium',
                 'Total phenols','Flavanoids',
                 'Nonflavanoid phenols','Proanthocyanins',
                 'Color intensity','Hue','OD280/OD315 of diluted wines',
                 'Proline']
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.fit_transform(X_test)
'''
lr=LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)
print('Training accuracy:',lr.score(X_train_std, y_train))
print('Test accuracy:',lr.score(X_test_std,y_test))
print(lr.intercept_)
print(lr.coef_)
'''
'''
ax=plt.subplot(111)
colors=['blue','green','red','cyan','magenta','yellow',
        'black','pink','lightgreen','lightblue','gray','indigo','orange']
weights,params=[],[]
for c in np.arange(-4,6):
    lr=LogisticRegression(penalty='l1',C=10**c,random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights=np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],
             label=df_wine.columns[column+1],
             color=color)
plt.axhline(0, color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=False)
plt.show()
'''
'''
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,
                 test_size=0.25,random_state=1):
        self.scoring=scoring
        self.estimator=clone(estimator)
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
    def fit(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,
            test_size=self.test_size,random_state=self.random_state)
        dim=X_train.shape[1]
        self.indices_=tuple(range(dim))
        self.subsets_=[self.indices_]
        score=self._calc_score(X_train,y_train,X_test,y_test,
                               self.indices_)
        self.scores_=[score]
        while dim>self.k_features:
            scores=[]
            subsets=[]
            
            for p in combinations(self.indices_,r=dim-1):
                score=self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best=np.argmax(scores)
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim-=1
            
            self.scores_.append(scores[best])
        self.k_score_=self.scores_[-1]
        return self
    def transform(self,X):
        return X[:,self.indices_]
    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:,indices],y_train)
        y_pred=self.estimator.predict(X_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score
    
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn=KNeighborsClassifier(n_neighbors=2)
sbs=SBS(knn,k_features=1)
sbs.fit(X_train_std,y_train)

k_feat=[len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5=list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
'''

from sklearn.ensemble import RandomForestClassifier
feat_labels=df_wine.columns[1:]
forest=RandomForestClassifier(n_estimators=10000,
                        random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" %(f+1,30,feat_labels[f],
                           importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()



