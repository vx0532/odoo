import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.decomposition.tests.test_nmf import random_state

df=pd.read_csv('/home/caofa/odoo-dev/my_modules/BackUp/MachineLearning/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
            'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
X = df[['RM']].values
y = df['MEDV'].values
'''
import seaborn as sns
#sns.reset_orig()
sns.set(style='whitegrid',context='notebook')
cols=['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(df[cols],size=2.5)
#plt.show()
import numpy as np
cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm, cbar=True,annot=True,square=True,
               fmt='.2f',annot_kws={'size':15},
               yticklabels=cols,xticklabels=cols)
plt.show()
'''
'''
class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            print(errors)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)
        
X = df[['RM']].values
y = df['MEDV'].values
def lin_regplot(X,y,model):
    plt.scatter(X,y,c='blue')
    plt.plot(X,model.predict(X),color='red')
    return None
'''
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000 \'s [MEDV] (standardized)')
plt.show()
'''
'''
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(X,y)
print('Slope:%.3f'% slr.coef_[0])
print('Intercept:%.3f'%slr.intercept_)
lin_regplot(X,y,slr)
plt.show()
'''
'''
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
ransac=RANSACRegressor(LinearRegression(),
            max_trials=100,
            min_samples=50,
            residual_metric=lambda x:np.sum(np.abs(x),axis=1), 
            residual_threshold=5.0,
            random_state=0)
ransac.fit(X, y)
inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],
            c='lightgreen',marker='s',label='Outliers')
plt.plot(line_X,line_y_ransac,color='red')
plt.xlabel('Average nunber of rooms [RM')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')

plt.show()
'''
'''
from sklearn.cross_validation import train_test_split
X=df.iloc[:,:-1].values 
y=df['MEDV'].values 
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=0)
slr=LinearRegression()
slr.fit(X_train,y_train)
y_train_pred=slr.predict(X_train)
y_test_pred=slr.predict(X_test)
plt.scatter(y_train_pred, y_train_pred-y_train, 
        c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,
        c='lightgreen',marker='s',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50,linewidth=2,colors='red')
plt.xlim([-10,50])

from sklearn.metrics import mean_squared_error
print('MSE train:%.3f,test:%.3f' %(
    mean_squared_error(y_train,y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
'''
X=np.array([258.0,270.0,294.0,320.0,342.0,368.0,
            396.0,446.0,480.0,586.0])[:,np.newaxis]
y=np.array([236.4,234.4,252.8,298.6,314.2,342.2,
            360.8,368.0,391.2,390.8])
lr=LinearRegression()
pr=LinearRegression()
quadratic=PolynomialFeatures(degree=2)
X_quad=quadratic.fit_transform(X)
lr.fit(X,y)
X_fit=np.arange(250,600,10)[:,np.newaxis]
y_lin_fit=lr.predict(X_fit)
pr.fit(X_quad,y)
y_quad_fit=pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X,y,label='training points')
plt.plot(X_fit,y_lin_fit,label='linear fit',linestyle='--')
plt.plot(X_fit,y_quad_fit,label='quadratic fit')
plt.legend(loc='upper left')
#plt.show()
y_lin_pred=lr.predict(X)
y_quad_pred=pr.predict(X_quad)
print('Training MSE linear linear: %.3f,quadratic:%.3f' 
      %(mean_squared_error(y,y_lin_pred),
        mean_squared_error(y,y_quad_pred)))
print('Traing R^2 linear: %.3f, quadratic:%.3f' 
      %(r2_score(y,y_lin_pred),
        r2_score(y,y_quad_pred)) )
'''
X=df[['LSTAT']].values 
y=df['MEDV'].values
regr=LinearRegression()
'''
quadratic=PolynomialFeatures(degree=2)
cubic=PolynomialFeatures(degree=3)
X_quad=quadratic.fit_transform(X)
X_cubic=cubic.fit_transform(X)
X_fit=np.arange(X.min(),X.max(),1)[:,np.newaxis]
regr=regr.fit(X, y)
y_lin_fit=regr.predict(X_fit)
linear_r2=r2_score(y, regr.predict(X))

regr=regr.fit(X_quad,y)
y_quad_fit=regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2=r2_score(y, regr.predict(X_quad))

regr=regr.fit(X_cubic,y)
y_cubic_fit=regr.predict(cubic.fit_transform(X_fit))
cubic_r2=r2_score(y,regr.predict(X_cubic))

plt.scatter(X,y,label='training points',color='lightgray')
plt.plot(X_fit,y_lin_fit,label='linear(d=1),$R^2=%.2f$' %linear_r2,
         color='blue',lw=2,linestyle=":")
plt.plot(X_fit,y_quad_fit,label="quadratic(d=2),$R^2=%.2f$" 
         % quadratic_r2,color='red',lw=2,linestyle='-')
plt.plot(X_fit,y_cubic_fit,label='cubic(d=3),$R^2=%.2f$'
         %cubic_r2,color='green',lw=2,linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()
'''
'''
X_log=np.log(X)
y_sqrt=np.sqrt(y)
X_fit=np.arange(X_log.min()-1,
                X_log.max()+1,1)[:,np.newaxis]
regr=regr.fit(X_log,y_sqrt)
y_lin_fit=regr.predict(X_fit)
linear_r2=r2_score(y_sqrt, regr.predict(X_log))

plt.scatter(X_log,y_sqrt,label='training points',
            color='lightgray')
plt.plot(X_fit,y_lin_fit,label='linear (d=1), $R^2=%.2f$'
         %linear_r2,color='blue',lw=2)
plt.xlabel('log(%lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{ Price \; in \; \$1000\'s [MEDV]}$')
plt.legend(loc='lower left')
plt.show()
'''
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None
'''
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)
sort_idx=X.flatten().argsort()
lin_regplot(X[sort_idx],y[sort_idx],tree)
plt.xlabel('%lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()
'''
X=df.iloc[:,:-1].values 
y=df['MEDV'].values 
X_train,X_test,y_train,y_test=train_test_split(X,y,
                        test_size=0.4,random_state=1)
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=1000,criterion='mse',
                             random_state=1,n_jobs=-1)
forest.fit(X_train,y_train)
y_train_pred=forest.predict(X_train)
y_test_pred=forest.predict(X_test)
print('MSE train: %.3f,test:%.3f' 
      %(mean_squared_error(y_train,y_train_pred),
        mean_squared_error(y_test,y_test_pred)))
print('R^2 train:%.3f$,test:%.3f'
      %(r2_score(y_train,y_train_pred),
        r2_score(y_test,y_test_pred)))

plt.scatter(y_train_pred,y_train_pred-y_train,c='black',
            marker='o',s=35,alpha=0.5,label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',
            marker='s',s=35,alpha=0.7,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()










