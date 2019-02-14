"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-13
 Time: 오전 10:34
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

# Params
Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 2000

# Launch
reward_list = list()

print("Training started ...")
for i in range(num_episodes):
    state = env.reset()
    reward_total = 0

    while True:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))  # with noise

        next_state, reward, done, info = env.step(action)

        # current_state 의 행동을 일정 수준(1 - learning_rate) 만큼 유지하면서 next_state 의 행동을 (learning_rate) 만큼 받아들인다.
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
                           + learning_rate * (reward + dis * np.max(Q[next_state, :]))

        reward_total += reward
        state = next_state

        # End condition
        if done:
            if i % 100 == 0:
                print("{} th episode finished with reward : {}".format(i, reward))

            break

    reward_list.append(reward_total)

print("Training finished ...")

print("Success rate :", np.sum(reward_list) / len(reward_list))
print("Final Q-Table values :")
print("LEFT DOWN RIGHT UP")
print(Q)

# Show graph with rewards
plt.bar(range(len(reward_list)), reward_list, label="Success")
plt.legend()
plt.show()


"""
Training started ...
0 th episode finished with reward : 0.0
100 th episode finished with reward : 0.0
200 th episode finished with reward : 1.0
300 th episode finished with reward : 1.0
400 th episode finished with reward : 1.0
500 th episode finished with reward : 1.0
600 th episode finished with reward : 1.0
700 th episode finished with reward : 1.0
800 th episode finished with reward : 0.0
900 th episode finished with reward : 0.0
1000 th episode finished with reward : 0.0
1100 th episode finished with reward : 0.0
1200 th episode finished with reward : 1.0
1300 th episode finished with reward : 1.0
1400 th episode finished with reward : 0.0
1500 th episode finished with reward : 1.0
1600 th episode finished with reward : 1.0
1700 th episode finished with reward : 0.0
1800 th episode finished with reward : 1.0
1900 th episode finished with reward : 0.0
Training finished ...

Success rate : 0.6695
Final Q-Table values :
LEFT DOWN RIGHT UP
[[5.13339418e-01 1.34570730e-02 9.88351996e-03 2.99497207e-02]
 [1.87953956e-03 4.59316473e-03 2.97186995e-02 6.00071814e-01]
 [4.63991507e-01 4.54695162e-03 6.41438527e-03 2.90521531e-02]
 [4.66196840e-03 4.51435770e-04 4.41596088e-04 2.91487694e-02]
 [6.69513075e-01 6.94413719e-03 9.21400812e-03 8.58517710e-03]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.29939426e-05 2.00025025e-06 2.74622731e-03 3.33734375e-05]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.77126057e-03 0.00000000e+00 9.20393945e-03 7.82032060e-01]
 [0.00000000e+00 8.07869563e-01 2.58322711e-03 0.00000000e+00]
 [8.95284329e-01 3.90399423e-04 5.11266347e-04 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 4.10743492e-03 9.39349960e-01 0.00000000e+00]
 [0.00000000e+00 9.95009554e-01 2.58816946e-02 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
"""
