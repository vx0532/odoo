import numpy as np
from pandas.tseries.frequencies import Resolution
from IPython.core.oinspect import Colors
class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=10):
        self.eta=eta
        self.n_iter=n_iter
    def fit(self,X,y):
        self.w_=np.zeros(1+X.shape[1])
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] +=update*xi
                self.w_[0] +=update
                errors +=int(update!=0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) +self.w_[0]
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)

import pandas as pd
df=pd.read_csv('~/odoo-dev/my_modules/BackUp/Machine_Learning/iris.data', header=None)
import matplotlib.pyplot as plt

y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',1,-1)
X=df.iloc[0:100,[0,2]].values

ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X, y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[])



