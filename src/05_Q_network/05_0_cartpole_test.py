"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-15
 Time: 오전 10:08
"""

import gym

env = gym.make('CartPole-v0')

env.reset()
current_episode = 0
reward_total = 0

while current_episode < 10:
    env.render()

    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    print(next_state, reward, done)

    reward_total += reward

    if done:
        current_episode += 1

        print(current_episode, "th Reward for this episode was:", reward_total)

        reward_total = 0
        env.reset()


