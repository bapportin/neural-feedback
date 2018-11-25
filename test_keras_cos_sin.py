from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 


if __name__=="__main__":
    X=[]
    Y=[]
    for i in range(256):

        X.append(np.zeros(1))
        X[-1][0]=(2*i*np.pi)/256
        Y.append(np.zeros(2))
        Y[-1][0]=np.sin(X[-1][0])
        Y[-1][1]=np.cos(X[-1][0])
    X=np.array(X)
    Y=np.array(Y)
    model = Sequential()
    model.add(Dense(512, input_dim=1))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('tanh'))
    sgd = SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(X, Y, nb_epoch=10000)
    y1=model.predict(X)
    for i in range(256):
        print X[i],y1[i],Y[i],Y[i]-y1[i]
