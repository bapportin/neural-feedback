import layers
import activation
import context
import errconvert
import numpy as np
import models
import gym

def mkOneHot(x,size):
    ret=np.zeros(size)
    ret[x]=1
    return ret

def print_stat(*args):
    pass

def print_tab(m):
    print
    for i in range(5):
        X0=mkOneHot(i,5)
        print X0,m.predict(X0)    

def test_000_nchain():
    num_episodes=1000
    y = 0.99
    eps = 1
    lr = 0.2
    decay_factor = 0.99
    
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(5,64))
    m.add(layers.FullyConnectedLayer(64,2))
    env=gym.make("NChain-v0")
    env.env.slip=0.0
    acnt=[0,0]
    ac2=[0,0]
    for episode in xrange(num_episodes):
        print acnt,float(acnt[0])/max(1,acnt[1]),ac2
        print_tab(m)
        s = env.reset()
        done = False
        memory=[]
        steps=0
        while not done:
            steps+=1
            # select the action with highest cummulative reward
            if np.random.random() < eps or episode<30:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(m.predict(mkOneHot(s,5)))
            acnt[a]+=1
            #f=float(acnt[0])/max(1,acnt[1])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            X0=mkOneHot(s,5)
            X1=mkOneHot(new_s,5)
            Y0=m.predict(X0)
            Y1=m.predict(X1)
            #if acnt[a]>=acnt[1-a]:
            #    Y0[a]=Y0[a]*0.01+0.99*(r+y*np.max(Y1))
            #else:
            Y0[a]=r+y*np.max(Y1)
            if ac2[a]<=ac2[1-a]:
                m.fit([X0],[Y0],epochs=1,lr=0.7,print_stat=print_stat)
                ac2[a]+=1
            s=new_s
        if episode%10==0:
            eps *= decay_factor
        print steps

