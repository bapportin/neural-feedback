import layers
import activation
import context
import errconvert
import numpy as np
import models
import gym
import time

def mkOneHot(x,size):
    ret=np.zeros(size)
    ret[x]=1
    return ret

if __name__=="__main__":
    env=gym.make("FrozenLake-v0")
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(env.observation_space.n,64,convert=errconvert.linearResample))
    m.add(layers.FullyConnectedLayer(64,env.action_space.n,activation=activation.Sigmoid()))
    e=1.0
    dis=0.9
    lr=0.2
    episode=0
    
    while True:
        observation=env.reset()
        total_reward=0
        memory=[]
        while True:
            actions=m.predict(mkOneHot(observation,env.observation_space.n))
            #print (observation,actions)
            if np.random.rand(1)<e:
                a=env.action_space.sample()
            else:
                #print actions
                a=actions.argmax()
            #print a
            new_state,reward,done,_=env.step(a)
            total_reward+=reward
            #if reward>0:
            #    print new_state,reward,done,_
            #    env.render()
            #    time.sleep(1)
            memory.append((observation,a,new_state,reward))
            observation=new_state
            if done:
                if total_reward>0:
                    print (episode,total_reward,len(memory),e)
                    env.render()
                episode+=1                
                break
        r=0
        e=e*0.999
        for ostate,a,nstate,reward in memory:
            r=reward+r*dis
            X0=np.zeros(env.observation_space.n)
            X0[ostate]=1
            Y0=m.predict(X0)
            X1=np.zeros(env.observation_space.n)
            X1[nstate]=1
            Y1=m.predict(X1)
            y=Y0.copy()
            y[a]=0.9*y[a]+0.1*(0.001*reward+dis*Y1.max())
            #print (X0,y,Y0,Y1)
            m.fit([(X0,y)])
        if total_reward>0:
            for s in range(env.observation_space.n):
                X=mkOneHot(observation,env.observation_space.n)
                print s,m.predict(X)
