import layers
import activation
import context
import errconvert
import numpy as np
import models


if __name__=="__main__":
    DATA=[
        (np.array([0,0]),np.array([1,0,0.5])),
        (np.array([0,1]),np.array([0,1,0.7])),
        (np.array([1,0]),np.array([0,1,0.9])),
        (np.array([1,1]),np.array([1,0,0.2])),
        ]
    ctx=context.Context()
    l0=layers.FullyConnectedLayer(2,11)
    l1=layers.FullyConnectedLayer(11,3)
    lr0=lr1=0.5
    for i in range(1000):
        for X,y in DATA:
            h=l0.forward(X,0,ctx)
            r=l1.forward(h,1,ctx)
            e=y-r
            l1.backward(e,lr0,1,ctx)
            l0.backward(e,lr0,0,ctx)
            print (i,X,y,r,e,np.sum(e))
