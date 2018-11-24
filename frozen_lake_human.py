import gym
import time


if __name__=="__main__":
    env=gym.make("FrozenLake-v0")
    while True:
        env.reset()
        env.render()
        done=False
        while not done:
            a=input("an action in: "+str(env.action_space.n))
            new_state,reward,done,_=env.step(a)
            env.render()
        
