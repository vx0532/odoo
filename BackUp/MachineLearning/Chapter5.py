'''
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression


df_wine=pd.read_csv('~/odoo-dev/my_modules/BackUp/Machine_Learning/wine.data',header=None)
X,y=df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                    test_size=0.3,random_state=0)
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)
'''
from bokeh.charts.attributes import marker, color
from IPython.core.pylabtools import figsize
from statsmodels.sandbox.formula import Factor
'''
cov_mat=np.cov(X_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
tot=sum(eigen_vals)
var_exp=[i/tot for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
'''
'''
import matplotlib.pyplot as plt
plt.bar(range(1,14),var_exp,alpha=0.9,align='center',
        label='individual explained variance')
plt.step(range(1,14), cum_var_exp,where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
'''
'''
eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i
             in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w=np.hstack((eigen_pairs[0][1][:,np.newaxis],
            eigen_pairs[1][1][:,np.newaxis]))
X_train_pca=X_train_std.dot(w)
colors=['r','b','g']
markers=['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],
                c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
'''
'''
from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8,
                    c=cmap(idx),marker=markers[idx],label=cl)
'''
'''
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA 
pca=PCA(n_components=None)
lr=LogisticRegression()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
print(pca.explained_variance_ratio_)
'''
'''
np.set_printoptions(precision=4)
mean_vecs=[]
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], 
                             axis=0))

d=13
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    class_scatter=np.cov(X_train_std[y_train==label].T)
    S_W +=class_scatter

mean_overall=np.mean(X_train_std,axis=0)
d=13
S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n=X[y==i+1,:].shape[0]
    mean_vec=mean_vec.reshape(d,1)
    mean_overall=mean_overall.reshape(d,1)
    S_B +=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
eigen_vals,eigen_vecs=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) 
             for i in range(len(eigen_vals))]
eigen_pairs=sorted(eigen_pairs,key=lambda k:k[0],reverse=True)
'''
'''
tot=sum(eigen_vals.real)
discr=[(i/tot) for i in sorted(eigen_vals.real,reverse=True)]
cum_discr=np.cumsum(discr)
plt.bar(range(1,14),discr,alpha=0.5,align='center',
        label='individual "discriminability"')
plt.step(range(1,14),cum_discr,where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best')
plt.show()
'''
'''
w=np.hstack((eigen_pairs[0][1][:,np.newaxis].real,
             eigen_pairs[1][1][:,np.newaxis].real))

X_train_lda=X_train_std.dot(w)
colors=['r','b','g']
markers=['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0],
                X_train_lda[y_train==l,1],
                c=c,label=l,marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()
'''
'''
from sklearn.lda import LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_std, y_train)
lr=LogisticRegression()
lr=lr.fit(X_train_lda, y_train)
'''
'''
plot_decision_regions(X_train_lda,y_train,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
X_test_lda=lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
'''

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X,gamma,n_components):
    sq_dists=pdist(X,'sqeuclidean')
    mat_sq_dists=squareform(sq_dists)
    K=exp(-gamma*mat_sq_dists)
    N=K.shape[0]
    one_n=np.ones((N,N))/N
    K=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    eigvals,eigvecs=eigh(K)
    alphas=np.column_stack((eigvecs[:,-i] 
                    for i in range(1,n_components+1)))
    lambdas=[eigvals[-i] for i in range(1,n_components+1)]
    return alphas,lambdas

from sklearn.datasets import make_moons  
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
X,y=make_moons(n_samples=100, random_state=123)
scikit_kpca=KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_skernpca=scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1],
            color='red',marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],
            color='blue',marker='o',alpha=0.5)
plt.show()


