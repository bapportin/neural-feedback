import keras
import numpy as np

if __name__=="__main__":
    DATA=[
        (np.array([0,0]),np.array([1,0,0.5])),
        (np.array([0,1]),np.array([0,1,0.7])),
        (np.array([1,0]),np.array([0,1,0.9])),
        (np.array([1,1]),np.array([1,0,0.2])),
        ]
    m=keras.models.Sequential([
      
        keras.layers.Dense(11,input_dim=2),
        keras.layers.Activation('sigmoid'),
        keras.layers.Dense(3),
        keras.layers.Activation('sigmoid')])

    o=keras.optimizers.SGD(lr=0.7)

    m.compile(loss='binary_crossentropy', optimizer=o)
    
    X=np.array(map(lambda x: x[0],DATA))
    Y=np.array(map(lambda x: x[1],DATA))
    print X,Y
    #for i in range(200):
        #for X,Y in DATA:
    m.fit(X,Y,epochs=162)
    y=m.predict(X)
    for i in range(4):
        print X[i],y[i],Y[i]
    
