import layers
import activation
import context
import errconvert
import numpy as np
import models


if __name__=="__main__":
    DATA=[
        (np.array([-1,-1]),np.array([1,-1,0.5])),
        (np.array([-1,1]),np.array([-1,1,0.7])),
        (np.array([1,-1]),np.array([-1,1,0.9])),
        (np.array([1,1]),np.array([1,-1,0.2])),
        ]
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(2,32))
    m.add(layers.FullyConnectedLayer(32,3))
    m.fit(DATA,epochs=200,lr=0.1)
    for X,y in DATA:
        print X,y,m.predict(X)
