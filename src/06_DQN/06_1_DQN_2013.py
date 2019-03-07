"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-20
 Time: 오전 11:30
"""

import gym
import tensorflow as tf
import numpy as np

import random
from collections import deque

env = gym.make("CartPole-v0")


class MyDQN2013(object):

    def __init__(self, session, input_size, output_size, learning_rate=1e-3, name="DQN2013"):
        self.sess = session
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.name = name

        self._X = tf.placeholder(tf.float32, shape=[None, self.input_size])
        self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
        self._hypothesis = None
        self._loss = None
        self._optimizer = None

    def build_model(self, hidden_layer_size=10) -> None:
        with tf.variable_scope("Layer1"):
            W1 = tf.get_variable(name="W", shape=[self.input_size, hidden_layer_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name="b")
            logits1 = tf.matmul(self._X, W1) + b1
            layer1 = tf.nn.relu(logits1)

        with tf.variable_scope("Layer2"):
            W2 = tf.get_variable(name="W", shape=[hidden_layer_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([self.output_size]), name="b")
            logits2 = tf.matmul(layer1, W2) + b2

        self._hypothesis = logits2

        self._loss = tf.reduce_mean(tf.square(self._hypothesis - self._Y))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

    def train(self, x_train, y_train):
        return self.sess.run(fetches=[self._loss, self._optimizer], feed_dict={self._X: x_train, self._Y: y_train})

    def predict(self, x_test):
        return self.sess.run(fetches=self._hypothesis, feed_dict={self._X: x_test})


def train_minibatch(dqn: MyDQN2013, train_batch, discount_rate=.99):
    x_stack = np.empty(0).reshape(0, dqn.input_size)  # shape 을 맞춰준 빈 ndarray 생성
    y_stack = np.empty(0).reshape(0, dqn.output_size)

    for state, next_state, action, reward, done, info in train_batch:
        q_state = dqn.predict(state)

        if done:
            q_state[0, action] = reward
        else:
            q_state[0, action] = reward + discount_rate * np.max(dqn.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, q_state])

    return dqn.train(x_train=x_stack, y_train=y_stack)


def main():
    # Params
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    discount_rate = .99
    learning_rate = 1e-3
    buffer_memory = 50000
    num_episodes = 5000
    batch_size = 64
    step_history_size = 30  # 100
    victory_step_condition = 100.  # 200.

    train_buffer_queue = deque(maxlen=buffer_memory)  # maxlen 을 설정하면 queue 가 꽉 찬 상태일 때 맨 처음 item 이 자동 삭제된다.
    step_history_queue = deque(maxlen=step_history_size)

    avg_step_list = list()
    end_switch = False

    # Launch
    with tf.Session() as sess:
        dqn = MyDQN2013(session=sess, input_size=input_size, output_size=output_size, learning_rate=learning_rate)
        dqn.build_model(hidden_layer_size=10)

        sess.run(tf.global_variables_initializer())

        # Training
        print("Training started ...")
        for i in range(num_episodes):
            e = 1. / ((i / 10) + 1)
            step_count = 0

            state = env.reset()

            while True:
                action = None
                step_count += 1
                state_reshaped = np.reshape(state, [-1, input_size])

                # Choose the action affected by e-greedy
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(dqn.predict(x_test=state_reshaped))

                next_state, reward, done, info = env.step(action)

                # If fallen, give the penalty
                if done:
                    reward = -1

                # Update train data to the buffer
                next_state_reshaped = np.reshape(next_state, [-1, input_size])
                train_buffer_queue.append((state_reshaped, next_state_reshaped, action, reward, done, info))

                state = next_state

                # If the buffer is ready, train minibatch data with random state
                if len(train_buffer_queue) > batch_size:
                    train_minibatch_data = random.sample(train_buffer_queue, batch_size)
                    loss, _ = train_minibatch(dqn=dqn, train_batch=train_minibatch_data, discount_rate=discount_rate)

                # End condition
                if done:
                    print("{} th episode finished with steps : {}".format(i, step_count))

                    step_history_queue.append(step_count)

                    # End condition of victory
                    if len(step_history_queue) == step_history_queue.maxlen:
                        avg_step = np.mean(step_history_queue)

                        if i % 100 == 0:
                            avg_step_list.append(avg_step)
                            print("Current 'average step' of the 'step history queue' :", avg_step)

                        if avg_step > victory_step_condition:
                            print("Game cleared in {} th episode with average step : {}".format(i, avg_step))

                            end_switch = True

                    break

            # Check the end switch in episode loop
            if end_switch:
                break

        print("Average step history list :")
        print(avg_step_list)
        print("Training finished ...")

        # Testing
        state = env.reset()
        reward_total = 0

        while True:
            env.render()
            state_reshaped = np.reshape(state, [-1, input_size])

            action = np.argmax(dqn.predict(x_test=state_reshaped))

            next_state, reward, done, info = env.step(action)

            state = next_state
            reward_total += reward

            # End condition
            if done:
                print("Game finished with step :", reward_total)

                break


if __name__ == "__main__":
    main()


"""
Training started ...
0 th episode finished with steps : 12
1 th episode finished with steps : 10
2 th episode finished with steps : 15
3 th episode finished with steps : 21
4 th episode finished with steps : 11
5 th episode finished with steps : 13
6 th episode finished with steps : 9
7 th episode finished with steps : 10
8 th episode finished with steps : 14
9 th episode finished with steps : 17
10 th episode finished with steps : 10
11 th episode finished with steps : 10
12 th episode finished with steps : 11
13 th episode finished with steps : 11
14 th episode finished with steps : 13
15 th episode finished with steps : 12
16 th episode finished with steps : 19
17 th episode finished with steps : 19
18 th episode finished with steps : 12
19 th episode finished with steps : 11
20 th episode finished with steps : 8
21 th episode finished with steps : 9
22 th episode finished with steps : 10
23 th episode finished with steps : 8
24 th episode finished with steps : 8
25 th episode finished with steps : 11
26 th episode finished with steps : 10
27 th episode finished with steps : 16
28 th episode finished with steps : 11
29 th episode finished with steps : 9
30 th episode finished with steps : 8
31 th episode finished with steps : 9
32 th episode finished with steps : 10
33 th episode finished with steps : 10
34 th episode finished with steps : 10
35 th episode finished with steps : 14
36 th episode finished with steps : 11
37 th episode finished with steps : 12
38 th episode finished with steps : 10
39 th episode finished with steps : 11
40 th episode finished with steps : 8
41 th episode finished with steps : 29
42 th episode finished with steps : 15
43 th episode finished with steps : 11
44 th episode finished with steps : 15
45 th episode finished with steps : 11
46 th episode finished with steps : 10
47 th episode finished with steps : 10
48 th episode finished with steps : 11
49 th episode finished with steps : 8
50 th episode finished with steps : 14
51 th episode finished with steps : 9
52 th episode finished with steps : 11
53 th episode finished with steps : 10
54 th episode finished with steps : 12
55 th episode finished with steps : 9
56 th episode finished with steps : 10
57 th episode finished with steps : 9
58 th episode finished with steps : 11
59 th episode finished with steps : 9
60 th episode finished with steps : 11
61 th episode finished with steps : 10
62 th episode finished with steps : 12
63 th episode finished with steps : 10
64 th episode finished with steps : 9
65 th episode finished with steps : 9
66 th episode finished with steps : 13
67 th episode finished with steps : 10
68 th episode finished with steps : 9
69 th episode finished with steps : 9
70 th episode finished with steps : 10
71 th episode finished with steps : 12
72 th episode finished with steps : 9
73 th episode finished with steps : 10
74 th episode finished with steps : 10
75 th episode finished with steps : 11
76 th episode finished with steps : 12
77 th episode finished with steps : 18
78 th episode finished with steps : 12
79 th episode finished with steps : 12
80 th episode finished with steps : 11
81 th episode finished with steps : 10
82 th episode finished with steps : 10
83 th episode finished with steps : 9
84 th episode finished with steps : 8
85 th episode finished with steps : 11
86 th episode finished with steps : 10
87 th episode finished with steps : 9
88 th episode finished with steps : 10
89 th episode finished with steps : 12
90 th episode finished with steps : 11
91 th episode finished with steps : 11
92 th episode finished with steps : 11
93 th episode finished with steps : 13
94 th episode finished with steps : 13
95 th episode finished with steps : 14
96 th episode finished with steps : 10
97 th episode finished with steps : 18
98 th episode finished with steps : 15
99 th episode finished with steps : 16
100 th episode finished with steps : 11
Current 'average step' of the 'step history queue' : 11.633333333333333
101 th episode finished with steps : 13
102 th episode finished with steps : 12
103 th episode finished with steps : 12
104 th episode finished with steps : 14
105 th episode finished with steps : 19
106 th episode finished with steps : 15
107 th episode finished with steps : 21
108 th episode finished with steps : 14
109 th episode finished with steps : 17
110 th episode finished with steps : 15
111 th episode finished with steps : 15
112 th episode finished with steps : 20
113 th episode finished with steps : 15
114 th episode finished with steps : 17
115 th episode finished with steps : 18
116 th episode finished with steps : 34
117 th episode finished with steps : 14
118 th episode finished with steps : 22
119 th episode finished with steps : 32
120 th episode finished with steps : 64
121 th episode finished with steps : 29
122 th episode finished with steps : 50
123 th episode finished with steps : 16
124 th episode finished with steps : 15
125 th episode finished with steps : 11
126 th episode finished with steps : 13
127 th episode finished with steps : 12
128 th episode finished with steps : 9
129 th episode finished with steps : 10
130 th episode finished with steps : 9
131 th episode finished with steps : 10
132 th episode finished with steps : 9
133 th episode finished with steps : 11
134 th episode finished with steps : 10
135 th episode finished with steps : 11
136 th episode finished with steps : 10
137 th episode finished with steps : 9
138 th episode finished with steps : 10
139 th episode finished with steps : 9
140 th episode finished with steps : 10
141 th episode finished with steps : 11
142 th episode finished with steps : 10
143 th episode finished with steps : 9
144 th episode finished with steps : 9
145 th episode finished with steps : 11
146 th episode finished with steps : 12
147 th episode finished with steps : 11
148 th episode finished with steps : 11
149 th episode finished with steps : 11
150 th episode finished with steps : 11
151 th episode finished with steps : 10
152 th episode finished with steps : 10
153 th episode finished with steps : 10
154 th episode finished with steps : 14
155 th episode finished with steps : 10
156 th episode finished with steps : 11
157 th episode finished with steps : 10
158 th episode finished with steps : 10
159 th episode finished with steps : 12
160 th episode finished with steps : 11
161 th episode finished with steps : 12
162 th episode finished with steps : 29
163 th episode finished with steps : 22
164 th episode finished with steps : 20
165 th episode finished with steps : 44
166 th episode finished with steps : 35
167 th episode finished with steps : 26
168 th episode finished with steps : 49
169 th episode finished with steps : 22
170 th episode finished with steps : 21
171 th episode finished with steps : 20
172 th episode finished with steps : 14
173 th episode finished with steps : 19
174 th episode finished with steps : 38
175 th episode finished with steps : 20
176 th episode finished with steps : 21
177 th episode finished with steps : 25
178 th episode finished with steps : 25
179 th episode finished with steps : 18
180 th episode finished with steps : 29
181 th episode finished with steps : 22
182 th episode finished with steps : 53
183 th episode finished with steps : 16
184 th episode finished with steps : 24
185 th episode finished with steps : 17
186 th episode finished with steps : 19
187 th episode finished with steps : 29
188 th episode finished with steps : 70
189 th episode finished with steps : 47
190 th episode finished with steps : 31
191 th episode finished with steps : 33
192 th episode finished with steps : 30
193 th episode finished with steps : 45
194 th episode finished with steps : 36
195 th episode finished with steps : 27
196 th episode finished with steps : 32
197 th episode finished with steps : 26
198 th episode finished with steps : 28
199 th episode finished with steps : 55
200 th episode finished with steps : 25
Current 'average step' of the 'step history queue' : 29.8
201 th episode finished with steps : 30
202 th episode finished with steps : 27
203 th episode finished with steps : 86
204 th episode finished with steps : 46
205 th episode finished with steps : 99
206 th episode finished with steps : 47
207 th episode finished with steps : 38
208 th episode finished with steps : 30
209 th episode finished with steps : 33
210 th episode finished with steps : 36
211 th episode finished with steps : 45
212 th episode finished with steps : 39
213 th episode finished with steps : 28
214 th episode finished with steps : 45
215 th episode finished with steps : 46
216 th episode finished with steps : 49
217 th episode finished with steps : 52
218 th episode finished with steps : 80
219 th episode finished with steps : 30
220 th episode finished with steps : 49
221 th episode finished with steps : 41
222 th episode finished with steps : 58
223 th episode finished with steps : 30
224 th episode finished with steps : 25
225 th episode finished with steps : 24
226 th episode finished with steps : 29
227 th episode finished with steps : 38
228 th episode finished with steps : 43
229 th episode finished with steps : 36
230 th episode finished with steps : 50
231 th episode finished with steps : 33
232 th episode finished with steps : 58
233 th episode finished with steps : 44
234 th episode finished with steps : 41
235 th episode finished with steps : 36
236 th episode finished with steps : 44
237 th episode finished with steps : 44
238 th episode finished with steps : 67
239 th episode finished with steps : 29
240 th episode finished with steps : 31
241 th episode finished with steps : 36
242 th episode finished with steps : 38
243 th episode finished with steps : 39
244 th episode finished with steps : 42
245 th episode finished with steps : 37
246 th episode finished with steps : 33
247 th episode finished with steps : 51
248 th episode finished with steps : 80
249 th episode finished with steps : 42
250 th episode finished with steps : 52
251 th episode finished with steps : 86
252 th episode finished with steps : 37
253 th episode finished with steps : 37
254 th episode finished with steps : 53
255 th episode finished with steps : 45
256 th episode finished with steps : 43
257 th episode finished with steps : 69
258 th episode finished with steps : 77
259 th episode finished with steps : 61
260 th episode finished with steps : 46
261 th episode finished with steps : 50
262 th episode finished with steps : 43
263 th episode finished with steps : 86
264 th episode finished with steps : 33
265 th episode finished with steps : 39
266 th episode finished with steps : 59
267 th episode finished with steps : 61
268 th episode finished with steps : 47
269 th episode finished with steps : 40
270 th episode finished with steps : 103
271 th episode finished with steps : 62
272 th episode finished with steps : 101
273 th episode finished with steps : 35
274 th episode finished with steps : 32
275 th episode finished with steps : 94
276 th episode finished with steps : 63
277 th episode finished with steps : 54
278 th episode finished with steps : 34
279 th episode finished with steps : 45
280 th episode finished with steps : 86
281 th episode finished with steps : 37
282 th episode finished with steps : 61
283 th episode finished with steps : 37
284 th episode finished with steps : 87
285 th episode finished with steps : 51
286 th episode finished with steps : 39
287 th episode finished with steps : 52
288 th episode finished with steps : 68
289 th episode finished with steps : 85
290 th episode finished with steps : 47
291 th episode finished with steps : 58
292 th episode finished with steps : 41
293 th episode finished with steps : 62
294 th episode finished with steps : 85
295 th episode finished with steps : 40
296 th episode finished with steps : 82
297 th episode finished with steps : 66
298 th episode finished with steps : 35
299 th episode finished with steps : 42
300 th episode finished with steps : 50
Current 'average step' of the 'step history queue' : 57.7
301 th episode finished with steps : 71
302 th episode finished with steps : 53
303 th episode finished with steps : 55
304 th episode finished with steps : 74
305 th episode finished with steps : 64
306 th episode finished with steps : 99
307 th episode finished with steps : 46
308 th episode finished with steps : 51
309 th episode finished with steps : 135
310 th episode finished with steps : 80
311 th episode finished with steps : 72
312 th episode finished with steps : 68
313 th episode finished with steps : 46
314 th episode finished with steps : 49
315 th episode finished with steps : 49
316 th episode finished with steps : 50
317 th episode finished with steps : 68
318 th episode finished with steps : 51
319 th episode finished with steps : 67
320 th episode finished with steps : 124
321 th episode finished with steps : 67
322 th episode finished with steps : 75
323 th episode finished with steps : 67
324 th episode finished with steps : 63
325 th episode finished with steps : 119
326 th episode finished with steps : 66
327 th episode finished with steps : 62
328 th episode finished with steps : 67
329 th episode finished with steps : 73
330 th episode finished with steps : 52
331 th episode finished with steps : 74
332 th episode finished with steps : 59
333 th episode finished with steps : 78
334 th episode finished with steps : 76
335 th episode finished with steps : 64
336 th episode finished with steps : 52
337 th episode finished with steps : 67
338 th episode finished with steps : 82
339 th episode finished with steps : 77
340 th episode finished with steps : 61
341 th episode finished with steps : 59
342 th episode finished with steps : 66
343 th episode finished with steps : 83
344 th episode finished with steps : 64
345 th episode finished with steps : 75
346 th episode finished with steps : 74
347 th episode finished with steps : 92
348 th episode finished with steps : 129
349 th episode finished with steps : 106
350 th episode finished with steps : 112
351 th episode finished with steps : 124
352 th episode finished with steps : 126
353 th episode finished with steps : 123
354 th episode finished with steps : 111
355 th episode finished with steps : 113
356 th episode finished with steps : 125
357 th episode finished with steps : 128
358 th episode finished with steps : 123
359 th episode finished with steps : 119
360 th episode finished with steps : 121
361 th episode finished with steps : 127
362 th episode finished with steps : 131
363 th episode finished with steps : 136
364 th episode finished with steps : 129
365 th episode finished with steps : 146
Game cleared in 365 th episode with average step : 102.7
Average step history list :
[11.633333333333333, 29.8, 57.7]
Training finished ...

Game finished with step : 120.0
"""
