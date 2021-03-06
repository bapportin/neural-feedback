import layers
import activation
import context
import errconvert
import numpy as np
import models
import gym
import time
import random


if __name__=="__main__":
    env=gym.make("CartPole-v0")
    m=models.Sequential()
    m.add(layers.FullyConnectedLayer(env.observation_space.shape[0],256,activation=activation.Logabs()))
    #m.add(layers.FullyConnectedLayer(32,64,activation=activation.Sigmoid()))
    m.add(layers.FullyConnectedLayer(256,env.action_space.n,activation=activation.Logabs()))
    e=1.0
    dis=0.99
    episode=0
    lr=0.1
    while True:
        new_state=env.reset()
        memory=[]
        total_reward=0
        while True:
            observation=new_state
            #print "observation",observation,env.observation_space.n
            actions=m.predict(observation)
            #print observation,actions
            if np.random.rand(1)<e:
                a=env.action_space.sample()
            else:
                #print "actions",actions,env.action_space.n
                a=actions.argmax()
            #print a
            new_state,reward,done,_=env.step(a)
            total_reward+=reward
            #print new_state,reward,done
            #if reward>0:
            #    #print new_state,reward,done,_
            env.render()
            #time.sleep(1)
            memory.append((observation,a,new_state,reward))
            
            if done:
                print total_reward
                delay=600
                e=(200+delay-total_reward)/(200+max(episode,delay))
                lr=1.0/total_reward
                episode+=1
                break
        r1=r=0
        #e=e*0.999
        DATA=[]
        r=0
        random.shuffle(memory)
        XX=[]
        YY=[]
        for i,(ostate,a,nstate,reward) in enumerate(memory):
            
            #    reward=0
            r1+=reward
            X=ostate
            oy=m.predict(ostate)#np.zeros(env.action_space.n)
            ny=m.predict(nstate)
            Y=oy.copy()
            Y[a]=0.99*np.max(ny)+reward
            #if i+1==len(memory):
            #    Y[a]=reward
            #DATA.append((X,Y))
            XX.append(X)
            YY.append(Y)
            print X,Y,oy,ny,Y-oy
        X=memory[-1][2]
        Y=np.zeros_like(YY[-1])
        XX.append(X)
        YY.append(Y)
        print X,Y
        m.fit(XX,YY,epochs=10,lr=0.1)
        print e,dis,r1,episode,total_reward,lr
        
