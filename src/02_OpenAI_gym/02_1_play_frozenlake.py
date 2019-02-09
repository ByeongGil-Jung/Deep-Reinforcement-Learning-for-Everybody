"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-07
 Time: 오전 11:47
"""

import gym
from gym.envs.registration import register

import readchar

# Key
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    "\x1b[A": UP,
    "\x1b[B": DOWN,
    "\x1b[C": RIGHT,
    "\x1b[D": LEFT
}

# Register
register(
    id="FrozenLake-v3",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False}
)

env = gym.make("FrozenLake-v3")
env.render()  # Show the initial board

# Launch
while True:
    # Choose an action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print("Game aborted !")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()

    print("State: {}, Action : {}, Reward: {}, Info: {}".format(state, action, reward, info))

    if done:
        print("Finished with reward", reward)
        break
