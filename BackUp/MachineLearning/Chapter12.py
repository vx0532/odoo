import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path,kind='train'):
    labels_path=os.path.join(path,'%s-labels-idx1-ubyte'
                             % kind)
    images_path=os.path.join(path,'%s-images-idx3-ubyte'
                             % kind)
    with open(labels_path,'rb') as lbpath:
        magic,n=struct.unpack('>II', lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols=struct.unpack(">IIII",
                                          imgpath.read(16))
        images=np.fromfile(imgpath,
                dtype=np.uint8).reshape(len(labels),784)
    return images,labels
X_train,y_train=load_mnist('/home/caofa/odoo-dev/my_modules/BackUp/MachineLearning/mnist', 
                           kind='train')
fig,ax=plt.subplots(nrows=2, ncols=5, 
                    sharex=True, sharey=True)
ax=ax.flatten()
for i in range(10):
    img=X_train[y_train==i][0].reshape(28,28)















