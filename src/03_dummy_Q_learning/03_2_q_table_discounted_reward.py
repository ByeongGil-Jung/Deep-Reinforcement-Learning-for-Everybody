"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-08
 Time: 오후 2:32
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Register
register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False}
)

# Params
env = gym.make(id="FrozenLake-v3")

Q = np.zeros(shape=[env.observation_space.n, env.action_space.n])
dis = .99  # (== gamma)
num_episodes = 2000

# Launch
reward_list = list()

print("Training started ...")
for i in range(num_episodes):
    state = env.reset()
    reward_total = 0

    # Start game
    while True:
        """
        < Noise >
        
        - 노이즈를 추가함으로써 비슷한 value 를 가진 다양한 action 을 취하도록 함
          이 때, iteration 이 진행 될 수록 영향력이 점점 작아진다.
        
        - 또한, np.argmax 는 값이 같을 경우 [0] 을 반환하는데,
          뒤에 임의의 값을 더해줌으로써 중복된 값들이여도 순수하게 random 한 argmax 를 반환하도록 함
        """
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        next_state, reward, done, info = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[next_state, :])

        reward_total += reward
        state = next_state

        # End condition
        if done:
            if i % 100 == 0:
                print("{} th episode finished with reward : {}".format(i, reward))

            break

    reward_list.append(reward_total)

print("Training finished ...")

print("Success rate :", np.sum(reward_list) / num_episodes)
print("Final Q-Table values :")
print("LEFT DOWN RIGHT UP")
print(Q)

# Show graph of rewards
plt.plot(range(len(reward_list)), reward_list, label="Success")
plt.legend()
plt.show()


"""
Training started ...
0 th episode finished with reward : 0.0
100 th episode finished with reward : 1.0
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

Success rate : 0.9665
Final Q-Table values :
LEFT DOWN RIGHT UP
[[0.         0.95099005 0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.96059601 0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.970299   0.        ]
 [0.         0.9801     0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.         0.         0.99       0.        ]
 [0.         0.         1.         0.        ]
 [0.         0.         0.         0.        ]]
"""
