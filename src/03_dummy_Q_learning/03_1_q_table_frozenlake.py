"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-08
 Time: 오전 11:08
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

import random


# Numpy 는 argmax 할 때 값이 같은 경우 항상 [0] 의 value 를 반환한다.
# 값이 같은 경우(예: 0, 0, 0, 0) 에서의 random 을 보장하고자 하는 함수이다.
def rargmax(vector):
    _max = np.amax(vector)
    indices = np.nonzero(vector == _max)[0]

    return random.choice(indices)


# Register
register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False}
)

# Params
env = gym.make(id="FrozenLake-v3")

Q = np.zeros(shape=[env.observation_space.n, env.action_space.n])  # (게임 space 크기, 각 space 당 action 갯수) == (16, 4)
num_episodes = 2000  # == training_epochs

# Launch
reward_list = list()

print("Training started ...")
for i in range(num_episodes):
    state = env.reset()
    reward_total = 0

    # Start game
    while True:
        # 현재 state 에서 reward 의 기록을 갖는 (==1 을 갖는) action 을 찾는다. (없다면 random 한 index 를 반환한다.)
        action = rargmax(Q[state, :])

        # state 에서 new_state 의 정보를 얻는다.
        new_state, reward, done, info = env.step(action)

        # new_state 에서의 reward 기록을 따라 현재 state 의 action 의 값을 update 한다.
        Q[state, action] = reward + np.max(Q[new_state, :])

        reward_total += reward  # 만약 goal 에 도달했을 경우 reward (==1) 을 저장한다.
        state = new_state  # new_state 로 이동한다.

        # End condition
        if done:
            if i % 100 == 0:
                print(i, "th episode finished with reward :", reward)

            break

    reward_list.append(reward_total)

print("Training finished ...")

print("Success rate :", np.sum(reward_list) / num_episodes)
print("Final Q-Table value :")
print("LEFT DOWN RIGHT UP")
print(Q)

# Show graph of rewards
plt.plot(range(len(reward_list)), reward_list, label="Success")
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
800 th episode finished with reward : 1.0
900 th episode finished with reward : 1.0
1000 th episode finished with reward : 1.0
1100 th episode finished with reward : 1.0
1200 th episode finished with reward : 1.0
1300 th episode finished with reward : 1.0
1400 th episode finished with reward : 1.0
1500 th episode finished with reward : 1.0
1600 th episode finished with reward : 1.0
1700 th episode finished with reward : 1.0
1800 th episode finished with reward : 1.0
1900 th episode finished with reward : 1.0
Training finished ...

Success rate : 0.9425
Final Q-Table value :
LEFT DOWN RIGHT UP
[[0. 1. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 0.]]
"""
