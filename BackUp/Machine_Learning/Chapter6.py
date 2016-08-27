import pandas as pd
from astropy.io.fits.header import Header
df=pd.read_csv('~/odoo-dev/my_modules/BackUp/Machine_Learning/wdbc.data',header=None)
from sklearn.preprocessing import LabelEncoder
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=\
    train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
pipe_lr=Pipeline([('scl',StandardScaler()),
                  ('pca',PCA(n_components=2)),
                  ('clf',LogisticRegression(random_state=1))])
xx=pipe_lr.fit(X_train, y_train)
print(xx)
print('Test Accuracy:' % pipe_lr.score(X_test, y_test))








