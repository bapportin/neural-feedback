from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import gym

def mkOneHot(x,size):
    ret=np.zeros(size)
    ret[x]=1
    return ret

if __name__=="__main__":
    env=gym.make("NChain-v0")
    m=models.Sequential()
    m.add(Dense(64,input_shape=(5,)))
    m.add(Activation("relu"))
    m.add(Dense(2))
    m.add(Activation("relu"))
