import layers
import activation
import context
import errconvert
import numpy as np
import models



def test_000_xor():
    DATA=[
        (np.array([-1,-1]),np.array([1,-1,0.5])),
        (np.array([-1,1]),np.array([-1,1,0.7])),
        (np.array([1,-1]),np.array([-1,1,0.9])),
        (np.array([1,1]),np.array([1,-1,0.2])),
        ]
    X=map(lambda x: x[0],DATA)
    Y=map(lambda x: x[0],DATA)
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(2,32))
    m.add(layers.FullyConnectedLayer(32,3))
    m.fit(X,Y,epochs=200,lr=0.1)
    for X,y in DATA:
        y1=m.predict(X)  
        print X,y,y1,y-y1


def test_001_randmap():
    DATA=[]
    for i in range(4):
        DATA.append((
            np.random.randn(5),
            np.random.randn(3)
            ))
    X=map(lambda x: x[0],DATA)
    Y=map(lambda x: x[0],DATA)        
    m=models.Sequential()    
    m.add(layers.FullyConnectedLayer(5,64))
    m.add(layers.FullyConnectedLayer(64,3))
    m.fit(X,Y,epochs=2000,lr=0.01)
    for X,y in DATA:
        y1=m.predict(X)  
        print X,y,y1,y-y1


def test_001_sin_cos():
    X,Y=[],[]
    for i in range(256):
        X.append(np.zeros(1))
        Y.append(np.zeros(2))
        X[-1][0]=(2*i*np.pi)/256
        Y[-1][0]=np.sin(X[-1])
        Y[-1][1]=np.cos(X[-1])
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(1,256,activation=activation.Logabs()))
    m.add(layers.FullyConnectedLayer(256,2,activation=activation.Logabs()))
    m.fit(X,Y,epochs=50000,lr=0.001)
    for i,x in enumerate(X):
        y0=Y[i]
        y1=m.predict(x)  
        print x,y0,y1,y0-y1

