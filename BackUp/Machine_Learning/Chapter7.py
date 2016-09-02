'''
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
from astropy.constants.si import alpha
from bokeh.charts.attributes import marker
from bokeh.charts.builders.scatter_builder import Scatter
from sympy.physics.quantum.spin import Rotation



class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.named_classifiers={key:value for key,value in
                                _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights
    def fit(self,X,y):
        self.lablenc_=LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_=self.lablenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf=clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    def predict(self,X):
        if self.vote=='probability':
            maj_vote=np.argmax(self.predict_proba(X),axis=1)
        else:
            predictions=np.asarray([clf.predict(X) for clf in
                                    self.classifiers_]).T
            maj_vote=np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                    weights=self.weights)),axis=1,arr=predictions)   
        maj_vote=self.lablenc_.inverse_transform(maj_vote)
        return maj_vote                                 
    def predict_proba(self,X):
        probas=np.asarray([clf.predict_proba(X) 
                           for clf in self.classifiers_])
        avg_proba=np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    def get_params(self,deep=True):
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out=self.named_classifiers.copy()
            for name,step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' %(name,key)]=value
            return out

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris=datasets.load_iris()
X,y=iris.data[50:,[1,2]],iris.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,
                                               random_state=1)
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

clf1=LogisticRegression(penalty='l2',C=0.001,random_state=0)
clf2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1=Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3=Pipeline([['sc',StandardScaler()],['clf',clf3]])
clf_labels=['Logistic Regression','Decision Tree','KNN']
print('10-fold cross validation:\n')
for clf,label in zip([pipe1,clf2,pipe3],clf_labels):
    scores=cross_val_score(estimator=clf, X=X_train, y=y_train, 
                           cv=10,scoring='roc_auc')
mv_clf=MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])
clf_labels+=['Majority Voting']
all_clf=[pipe1,clf2,pipe3,mv_clf]
for clf,label in zip(all_clf,clf_labels):
    scores=cross_val_score(estimator=clf, X=X_train, y=y_train, 
                    scoring='roc_auc', cv=10)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
colors=['black','orange','blue','green']
linestyles=[':','--','-.','-']
for clf,label,clr,ls in zip(all_clf,clf_labels,colors,linestyles):
    y_pred=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    fpr,tpr,thresholds=roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc=auc(x=fpr,y=tpr)

sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
from itertools import product
x_min=X_train_std[:,0].min()-1
x_max=X_train_std[:,0].max()+1
y_min=X_train_std[:,1].min()-1
y_max=X_train_std[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                  np.arange(y_min,y_max,0.1))
f,axarr=plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',
                     figsize=(7,5))
for idx,clf,tt in zip(product([0,1],[0,1]),all_clf,clf_labels):
    clf.fit(X_train_std,y_train)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx[0],idx[1]].scatter(X_train_std[y_train==0,0],
                                 X_train_std[y_train==0,1],
                                 c='blue',marker='^',s=50)
    axarr[idx[0],idx[1]].scatter(X_train_std[y_train==1,0],
                                 X_train_std[y_train==1,1],
                                 c='red',marker='o',s=50)
    axarr[idx[0],idx[1]].set_title(tt)
plt.text(-3.5,-4.5,s='Septal width [standardized]',
         ha='center',va='center',fontsize=12)
plt.text(-11.8,4.5,s='Petal length [standardized]',
         ha='center',va='center',fontsize=12, rotation=90)
plt.show()

from sklearn.grid_search import GridSearchCV
params={'decisiontreeclassifier__max_depth':[1,2],
        'pipeline-1__clf__C':[0.001,0.1,100.0]}
grid=GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,
                  scoring='roc_auc')
grid.fit(X_train, y_train)
print('Best parameters: %s' %grid.best_params_)
'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from astropy.constants.si import alpha
from bokeh.core.properties import FontSizeSpec
from bokeh.charts.attributes import marker
df_wine=pd.read_csv('/home/caofa/odoo-dev/my_modules/BackUp/Machine_Learning/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid','Ash',
                 'Alcalinity of ash','Magnesium','Total phenols',
                 'Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                 'Color intensity','Hue','OD280/OD315 of diluted wines',
                 'Proline']
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
X=df_wine[['Alcohol','Hue']].values
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,
                            test_size=0.4,random_state=1)
'''
from sklearn.ensemble import BaggingClassifier
tree=DecisionTreeClassifier(criterion='entropy',max_depth=None)
bag=BaggingClassifier(base_estimator=tree,n_estimators=500,
                      max_samples=1.0,max_features=1.0,
                      bootstrap=True,bootstrap_features=False,
                      n_jobs=1,random_state=1)
from sklearn.metrics import accuracy_score
tree=tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
tree_train=accuracy_score(y_train, y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
bag=bag.fit(X_train, y_train)
y_train_pred=bag.predict(X_train)
y_test_pred=bag.predict(X_test)
bag_train=accuracy_score(y_train, y_train_pred)
bag_test=accuracy_score(y_test,y_test_pred)

import matplotlib.pyplot as plt
x_min=X_train[:,0].min()-1
x_max=X_train[:,0].max()+1
y_min=X_train[:,1].min()-1
y_max=X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                  np.arange(y_min,y_max,0.1))
f,axarr=plt.subplots(nrows=1, ncols=2, 
            sharex='col', sharey='row',figsize=(8,3))
for idx,clf,tt in zip([0,1],
                      [tree,bag],
                      ['Decision Tree','Bagging']):
    clf.fit(X_train,y_train)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0],
                       X_train[y_train==0,1],
                       c='blue',marker='^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='red',marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.text(10,-1.0,s='Hue',ha='center',va='center',fontsize=12)
plt.show()
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
tree=DecisionTreeClassifier(criterion='entropy',
                            max_depth=1)
ada=AdaBoostClassifier(base_estimator=tree,
    n_estimators=500,learning_rate=0.1,random_state=0)
tree=tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)

x_min=X_train[:,0].min()-1
x_max=X_train[:,0].max()+1
y_min=X_train[:,1].min()-1
y_max=X_train[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                  np.arange(y_min,y_max,0.1))
f,axarr=plt.subplots(1,2,sharex='col',sharey='row',
                     figsize=(8,3))
for idx,clf,tt in zip([0,1],[tree,ada],
                      ['Decision Tree','AdaBoost']):
    clf.fit(X_train,y_train)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0],
                        X_train[y_train==0,1],
                        c='blue',marker='^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='red',marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.text(10.0,-1.0,s='Hue',ha='center',va='center',
         fontsize=12)
plt.show()























        
        
        
        
        
        
        
        
        