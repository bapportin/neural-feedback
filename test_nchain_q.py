import numpy as np
import gym

def eps_greedy_q_learning_with_table(env, num_episodes=1000):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.2
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[s, a] = q_table[s, a]*(1-lr) + lr*(r + (y * np.max(q_table[new_s, :])))
            s = new_s
    return q_table



if __name__=="__main__":
    env=gym.make("NChain-v0")
    #env.env.slip=0.1
    qt=eps_greedy_q_learning_with_table(env)
    print qt
