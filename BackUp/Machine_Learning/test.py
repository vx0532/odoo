import matplotlib.pyplot as plt
import numpy as np

x=[1,2,3,4]
y=[4,7,9,8]
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,0.8,0.9])
l1,=ax.plot(x,y)
l2,=ax.plot(y,x)
fig.legend((l1,l2),['line1','line2'],'right',ncol=1)
#plt.axis('equal')
plt.show()