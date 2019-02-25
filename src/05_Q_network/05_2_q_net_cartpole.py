"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-15
 Time: 오전 10:03
"""

"""
앞의 FrozenLake 와 마찬가지로, 학습이 잘 안 될 것.

q_hat 이 q_actual 에 수렴하지 않고 분산됨.
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

# Params
learning_rate = 0.1
discount_rate = 0.9
num_episodes = 5000
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])

# Layers
W1 = tf.get_variable(name="W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([output_size]), name="b1")
logits1 = tf.matmul(X, W1) + b1

# Model
loss = tf.reduce_sum(tf.square(logits1 - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Launch
step_history = list()
step_count_list = list()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    print("Training started ...")
    for i in range(num_episodes):
        step_count = 0
        e = 1. / ((i / 10) + 1)

        state = env.reset()

        while True:
            action = None
            step_count += 1
            state_reshaped = np.reshape(state, [1, input_size])

            q_state = sess.run(fetches=logits1, feed_dict={X: state_reshaped})

            # Choose the action affected by e-greedy
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_state)

            next_state, reward, done, info = env.step(action)  # if not fallen, always return the reward 1.0

            # If fallen
            if done:
                q_state[0, action] = -100  # give a big penalty
            else:
                next_state_reshaped = np.reshape(next_state, [1, input_size])
                q_next_state = sess.run(fetches=logits1, feed_dict={X: next_state_reshaped})

                q_state[0, action] = reward + discount_rate * np.max(q_next_state)

            # Train our network using q_state and state
            sess.run(fetches=optimizer, feed_dict={X: state_reshaped, Y: q_state})

            state = next_state

            # End condition
            if done:
                step_count_list.append(step_count)

                if i % 100 == 0:
                    print("{} th episode finished with steps : {}".format(i, step_count))

                break

        step_history.append(step_count)

        # Success condition
        if len(step_history) > 10 and np.mean(step_history[-10:]) > 500:
            break

    # Testing
    state = env.reset()
    reward_total = 0

    while True:
        env.render()
        state_reshaped = np.reshape(state, [1, input_size])

        pred_state = sess.run(fetches=logits1, feed_dict={X: state_reshaped})
        action = np.argmax(pred_state)

        next_state, reward, done, info = env.step(action)

        reward_total += reward

        if done:
            print("Testing finished with total reward :", reward_total)
            break

# Show graph with steps
plt.plot(range(len(step_count_list)), step_count_list, label="Step_Count")
plt.legend()
plt.show()

"""
Training started ...
0 th episode finished with steps : 19
100 th episode finished with steps : 9
200 th episode finished with steps : 9
300 th episode finished with steps : 11
400 th episode finished with steps : 10
500 th episode finished with steps : 31
600 th episode finished with steps : 12
700 th episode finished with steps : 10
800 th episode finished with steps : 15
900 th episode finished with steps : 25
1000 th episode finished with steps : 40
1100 th episode finished with steps : 30
1200 th episode finished with steps : 17
1300 th episode finished with steps : 46
1400 th episode finished with steps : 9
1500 th episode finished with steps : 11
1600 th episode finished with steps : 15
1700 th episode finished with steps : 16
1800 th episode finished with steps : 11
1900 th episode finished with steps : 23
2000 th episode finished with steps : 13
2100 th episode finished with steps : 27
2200 th episode finished with steps : 38
2300 th episode finished with steps : 85
2400 th episode finished with steps : 27
2500 th episode finished with steps : 26
2600 th episode finished with steps : 125
2700 th episode finished with steps : 29
2800 th episode finished with steps : 21
2900 th episode finished with steps : 13
3000 th episode finished with steps : 14
3100 th episode finished with steps : 23
3200 th episode finished with steps : 24
3300 th episode finished with steps : 63
3400 th episode finished with steps : 28
3500 th episode finished with steps : 25
3600 th episode finished with steps : 18
3700 th episode finished with steps : 17
3800 th episode finished with steps : 38
3900 th episode finished with steps : 44
4000 th episode finished with steps : 38
4100 th episode finished with steps : 57
4200 th episode finished with steps : 21
4300 th episode finished with steps : 17
4400 th episode finished with steps : 14
4500 th episode finished with steps : 12
4600 th episode finished with steps : 10
4700 th episode finished with steps : 25
4800 th episode finished with steps : 26
4900 th episode finished with steps : 24

Testing finished with total reward : 9.0
"""
