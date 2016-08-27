import matplotlib.pyplot as plt
import numpy as np
'''
X=[1,2,3,4,5,6]
y=[2.5,3.51,4.45,5.52,6.47,7.51]
z1=np.polyfit(X, y, 1)
p1=np.poly1d(z1)
'''
def polyfit(x,y,degree):
    results={}
    coeffs=np.polyfit(x, y, degree)
    results['polynomial']=coeffs.tolist()
    p=np.poly1d(coeffs)
    yhat=p(x)
    ybar=np.sum(y)/len(y)
    ssreg=np.sum((yhat-ybar)**2)
    sstot=np.sum((y-ybar)**2)
    results['determination']=ssreg/sstot
    return results
x=[1,2,3,4,5,6]
y=[2.5,3.51,4.45,5.52,6.47,7.2]
z1=polyfit(x, y, 2) 
print(z1['polynomial'])
z2=np.poly1d(z1['polynomial'])
print(z2)
plt.plot(z2)
plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    