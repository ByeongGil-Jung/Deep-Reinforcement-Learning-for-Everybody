"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-14
 Time: 오전 10:49
"""

"""
예제에 따르면, 학습이 잘 안 될 것.
> 이것을 보완한 것이 딥마인드에서 발표한 DQN 이다.

(내 생각엔 layer 의 갯수가 너무 작았기 때문이라고 생각한다.
적어도 state * action 만큼의 weight 는 필요할 것이다.)
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_state_with_one_hot(state_):
    return np.identity(16)[state_:state_ + 1]


env = gym.make("FrozenLake-v0")

# Params
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
discount_rate = 0.99
num_episodes = 2000

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])

# Layers
W1 = tf.get_variable(name="W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([output_size]), name="b1")
logits1 = tf.matmul(X, W1) + b1

# Model

# reduce_mean 을 사용하면 0.2 ~ 0.3 의 acc 를 얻지만, reduce_sum 을 사용하면 0.4 의 acc 를 얻는다. 왜 그럴까.
loss = tf.reduce_sum(tf.square(Y - logits1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)  # 여기서 Adam 을 사용하면 동작하지 않음

# Launch
reward_list = list()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training started ...")
    for i in range(num_episodes):
        reward_total = 0
        e = 1. / ((i / 50) + 10)  # e-greedy

        state = env.reset()

        while True:
            action = None

            q_state = sess.run(fetches=logits1, feed_dict={X: get_state_with_one_hot(state)})

            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_state)

            next_state, reward, done, info = env.step(action)

            # Update q_state
            if done:
                q_state[0, action] = reward  # if agent go hole or goal
            else:
                q_next_state = sess.run(fetches=logits1, feed_dict={X: get_state_with_one_hot(next_state)})

                q_state[0, action] = reward + discount_rate * np.max(q_next_state)

            # Train our network using q_state and state
            sess.run(fetches=optimizer, feed_dict={X: get_state_with_one_hot(state), Y: q_state})

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

# Show graph with rewards
plt.bar(range(len(reward_list)), reward_list, label="Success")
plt.legend()
plt.show()


"""
Training started ...
0 th episode finished with reward : 0.0
100 th episode finished with reward : 0.0
200 th episode finished with reward : 0.0
300 th episode finished with reward : 0.0
400 th episode finished with reward : 1.0
500 th episode finished with reward : 1.0
600 th episode finished with reward : 1.0
700 th episode finished with reward : 1.0
800 th episode finished with reward : 1.0
900 th episode finished with reward : 1.0
1000 th episode finished with reward : 1.0
1100 th episode finished with reward : 0.0
1200 th episode finished with reward : 0.0
1300 th episode finished with reward : 0.0
1400 th episode finished with reward : 0.0
1500 th episode finished with reward : 0.0
1600 th episode finished with reward : 1.0
1700 th episode finished with reward : 0.0
1800 th episode finished with reward : 1.0
1900 th episode finished with reward : 1.0
Training finished ...

Success rate : 0.45
"""
