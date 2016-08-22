
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

x=np.array([[1.0,2.0],[2,4],[0.5,2.5]])
#x=[[1,2],[4,9],[6,6],[7,1],[9,5]]
x_mean=np.mean(x,axis=0)
xx=x-x_mean
plt.scatter(xx[:,0],xx[:,1])

sc=StandardScaler()
#x_std=sc.fit_transform(x)
x_std=x
colors=['r','b','g']
#for le in range(2,len(x)+1):
for le in range(len(x),len(x)+1):
    cov_mat=np.cov(x_std[0:le,:].T)
    eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
    xx=[i for i in range(12)]
    yy1=[-i*eigen_vecs[0][0]/eigen_vecs[1][0] for i in xx]
    yy2=[-i*eigen_vecs[0][1]/eigen_vecs[1][1] for i in xx]
    plt.plot(xx,yy1,c=colors[(le-2)%3])
    plt.plot(xx,yy2,c=colors[(le-2)%3])
    plt.axis('equal')
plt.show()

'''
xxx=[0,1,2,3,4,5,6,7,8,9,10]
r=14+148**(1/2)
yyy=[(2/(2-r))*xxx[i] for i in xxx]
plt.plot(xxx,yyy)
r=14-148**(1/2)
zzz=[(2/(2-r))*xxx[i] for i in xxx]
plt.plot(xxx,zzz)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

x=np.array([[1.0,2.0],[2.0,3.0],[3,4]])
plt.scatter(x[:,0],x[:,1])
sc=StandardScaler()
x_std=sc.fit_transform(x)
cov_mat=np.cov(x_std.T)
print(cov_mat)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print(eigen_vecs)
xx=[i for i in range(12)]
yy1=[-i*eigen_vecs[0][0]/eigen_vecs[0][1] for i in xx]
yy2=[-i*eigen_vecs[1][0]/eigen_vecs[1][1] for i in xx]
plt.plot(xx,yy1)
plt.plot(xx,yy2)    
plt.show()
'''