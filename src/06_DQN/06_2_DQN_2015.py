"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-02-27
 Time: 오후 3:17
"""

"""
[ 기존 DQN2013 과의 차이점 ]

앞의 DQN_2013 에선 "비교해야 하는 값 (==target)" 에 불안정성이 있었다.

기존 q_state 는 

> target 값 : q_state = reward + discount_rate * np.max(dqn.predict(next_state))
> 학습 : dqn.train(state, q_state)

의 과정에서 이뤄진다.

따라서 동일한 network (dqn) 에서 예측한 값을 학습시킬 때의 target 으로 사용한다.
즉, 예측한 q_state 값을 target 으로 삼았기 때문에 매번 target 이 변화한다.

다시 말해,
학습을 하면서 바뀐 network 가 next_state 에 영향을 미치면서,
예측을 비교해야 할 target 값(==q_state) 의 값 또한 바뀌게 된다는 이야기이다.
-> 이는 무한한 feedback loop 에 빠지게 된다. (== 발산한다.)
(마치 화살을 쐈는데 표적이 움직인 것과 같다.)

따라서 target_dqn 을 따로 만들어 network 를 두 개 만든다.
그리고 일정 주기만큼 target_dqn <- main_dqn 으로 업데이트 시킨다.
이를 통해 학습이 안정적으로 수렴하게 된다.
"""

import tensorflow as tf
import numpy as np
import gym

from collections import deque
import random

env = gym.make("CartPole-v0")


class MyDQN2015(object):

    def __init__(self, session, input_size, output_size, learning_rate=1e-3, name="DQN2015"):
        self.sess = session
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.name = name

        self.hypothesis = None
        self.loss = None
        self.optimizer = None

        self._X = tf.placeholder(tf.float32, shape=[None, self.input_size])
        self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])

    def build_model(self, hidden_layer_size=10):
        with tf.variable_scope(self.name):
            W1 = tf.get_variable(name="W1", shape=[self.input_size, hidden_layer_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name="b1")
            logits1 = tf.matmul(self._X, W1) + b1
            layer1 = tf.nn.relu(logits1)

            W2 = tf.get_variable(name="W2", shape=[hidden_layer_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([self.output_size]), name="b2")
            logits2 = tf.matmul(layer1, W2) + b2

        self.hypothesis = logits2

        self.loss = tf.reduce_mean(tf.square(self.hypothesis - self._Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, x_data, y_data):
        return self.sess.run(fetches=[self.loss, self.optimizer], feed_dict={self._X: x_data, self._Y: y_data})

    def predict(self, x_data):
        return self.sess.run(fetches=self.hypothesis, feed_dict={self._X: x_data})


def get_copied_var_ops(src_scope_name: str, dest_scope_name: str):
    op_list = list()

    src_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_list.append(tf.assign(ref=dest_var, value=src_var.value()))  # 맞나 확인 할 것

    """
    src_var 과 src_var.value() 는 아래와 같은 차이가 있다.
    
    src_var : <tf.Variable 'main/W1:0' shape=(4, 10) dtype=float32_ref>
    src_var.value() : Tensor("main/W1/read:0", shape=(4, 10), dtype=float32)
    
    즉, tf.Variable 에 Tensor 값을 assign 하는 의미이다.
    """

    return op_list


def train_minibatch(main_dqn: MyDQN2015, target_dqn: MyDQN2015, train_batch, discount_rate=.99):
    x_stack = np.empty(0).reshape(0, main_dqn.input_size)
    y_stack = np.empty(0).reshape(0, main_dqn.output_size)

    for state, next_state, action, reward, done, info in train_batch:
        q_state = main_dqn.predict(state)

        if done:
            q_state[0, action] = reward
        else:
            q_state[0, action] = reward + discount_rate * np.max(target_dqn.predict(x_data=next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, q_state])

    return main_dqn.train(x_data=x_stack, y_data=y_stack)


def main():
    # Params
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    learning_rate = 1e-3
    discount_rate = .99
    num_episodes = 5000
    batch_size = 64
    buffer_memory = 50000
    step_history_size = 30  # 100
    victory_step_condition = 100.  # 200.

    target_update_frequency = 10

    train_buffer_queue = deque(maxlen=buffer_memory)
    step_history_queue = deque(maxlen=step_history_size)

    avg_step_list = list()
    end_switch = False

    # Launch
    with tf.Session() as sess:
        # Create models
        main_dqn = MyDQN2015(session=sess, input_size=input_size, output_size=output_size,
                             learning_rate=learning_rate, name="main")
        target_dqn = MyDQN2015(session=sess, input_size=input_size, output_size=output_size,
                               learning_rate=learning_rate, name="target")

        main_dqn.build_model(hidden_layer_size=10)
        target_dqn.build_model(hidden_layer_size=10)

        # Init session
        sess.run(tf.global_variables_initializer())

        # Define the copy-ops function
        copied_ops = get_copied_var_ops(src_scope_name="main", dest_scope_name="target")

        sess.run(copied_ops)

        for i in range(num_episodes):
            e = 1. / ((i / 10) + 1)
            step_count = 0

            state = env.reset()

            while True:
                action = None
                state_reshaped = np.reshape(state, [-1, input_size])

                step_count += 1

                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(main_dqn.predict(x_data=state_reshaped))

                next_state, reward, done, info = env.step(action)

                if done:
                    reward = -1

                next_state_reshaped = np.reshape(next_state, [-1, input_size])
                train_buffer_queue.append((state_reshaped, next_state_reshaped, action, reward, done, info))

                state = next_state

                # If the buffer is ready, train minibatch data with radnom state
                if len(train_buffer_queue) > batch_size:
                    train_minibatch_data = random.sample(train_buffer_queue, batch_size)
                    loss, _ = train_minibatch(main_dqn=main_dqn, target_dqn=target_dqn,
                                              train_batch=train_minibatch_data, discount_rate=discount_rate)

                # Update the target network
                if step_count % target_update_frequency == 0:
                    sess.run(fetches=copied_ops)

                # End condition
                if done:
                    print("{} th episode finished with step : {}".format(i, step_count))

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

            action = np.argmax(main_dqn.predict(x_data=state_reshaped))

            next_state, reward, done, info = env.step(action)

            state = next_state
            reward_total += reward

            if done:
                print("Game finished with step :", reward_total)

                break


if __name__ == "__main__":
    main()


"""
0 th episode finished with step : 11
1 th episode finished with step : 44
2 th episode finished with step : 29
3 th episode finished with step : 9
4 th episode finished with step : 23
5 th episode finished with step : 9
6 th episode finished with step : 12
7 th episode finished with step : 10
8 th episode finished with step : 11
9 th episode finished with step : 12
10 th episode finished with step : 17
11 th episode finished with step : 19
12 th episode finished with step : 17
13 th episode finished with step : 11
14 th episode finished with step : 13
15 th episode finished with step : 12
16 th episode finished with step : 12
17 th episode finished with step : 22
18 th episode finished with step : 12
19 th episode finished with step : 11
20 th episode finished with step : 24
21 th episode finished with step : 24
22 th episode finished with step : 8
23 th episode finished with step : 10
24 th episode finished with step : 18
25 th episode finished with step : 14
26 th episode finished with step : 18
27 th episode finished with step : 13
28 th episode finished with step : 9
29 th episode finished with step : 10
30 th episode finished with step : 8
31 th episode finished with step : 12
32 th episode finished with step : 12
33 th episode finished with step : 13
34 th episode finished with step : 11
35 th episode finished with step : 12
36 th episode finished with step : 10
37 th episode finished with step : 12
38 th episode finished with step : 11
39 th episode finished with step : 11
40 th episode finished with step : 9
41 th episode finished with step : 8
42 th episode finished with step : 10
43 th episode finished with step : 10
44 th episode finished with step : 14
45 th episode finished with step : 11
46 th episode finished with step : 11
47 th episode finished with step : 10
48 th episode finished with step : 9
49 th episode finished with step : 12
50 th episode finished with step : 10
51 th episode finished with step : 10
52 th episode finished with step : 12
53 th episode finished with step : 13
54 th episode finished with step : 10
55 th episode finished with step : 10
56 th episode finished with step : 10
57 th episode finished with step : 10
58 th episode finished with step : 8
59 th episode finished with step : 15
60 th episode finished with step : 10
61 th episode finished with step : 11
62 th episode finished with step : 12
63 th episode finished with step : 13
64 th episode finished with step : 11
65 th episode finished with step : 9
66 th episode finished with step : 13
67 th episode finished with step : 8
68 th episode finished with step : 9
69 th episode finished with step : 11
70 th episode finished with step : 9
71 th episode finished with step : 17
72 th episode finished with step : 9
73 th episode finished with step : 10
74 th episode finished with step : 12
75 th episode finished with step : 11
76 th episode finished with step : 10
77 th episode finished with step : 11
78 th episode finished with step : 11
79 th episode finished with step : 13
80 th episode finished with step : 11
81 th episode finished with step : 11
82 th episode finished with step : 13
83 th episode finished with step : 10
84 th episode finished with step : 10
85 th episode finished with step : 9
86 th episode finished with step : 12
87 th episode finished with step : 12
88 th episode finished with step : 12
89 th episode finished with step : 9
90 th episode finished with step : 12
91 th episode finished with step : 11
92 th episode finished with step : 11
93 th episode finished with step : 10
94 th episode finished with step : 11
95 th episode finished with step : 10
96 th episode finished with step : 13
97 th episode finished with step : 13
98 th episode finished with step : 11
99 th episode finished with step : 12
100 th episode finished with step : 11
Current 'average step' of the 'step history queue' : 11.266666666666667
101 th episode finished with step : 12
102 th episode finished with step : 14
103 th episode finished with step : 15
104 th episode finished with step : 12
105 th episode finished with step : 16
106 th episode finished with step : 14
107 th episode finished with step : 13
108 th episode finished with step : 14
109 th episode finished with step : 10
110 th episode finished with step : 17
111 th episode finished with step : 17
112 th episode finished with step : 9
113 th episode finished with step : 17
114 th episode finished with step : 17
115 th episode finished with step : 11
116 th episode finished with step : 11
117 th episode finished with step : 10
118 th episode finished with step : 11
119 th episode finished with step : 9
120 th episode finished with step : 10
121 th episode finished with step : 9
122 th episode finished with step : 27
123 th episode finished with step : 20
124 th episode finished with step : 19
125 th episode finished with step : 9
126 th episode finished with step : 18
127 th episode finished with step : 20
128 th episode finished with step : 12
129 th episode finished with step : 10
130 th episode finished with step : 9
131 th episode finished with step : 10
132 th episode finished with step : 23
133 th episode finished with step : 36
134 th episode finished with step : 18
135 th episode finished with step : 23
136 th episode finished with step : 27
137 th episode finished with step : 21
138 th episode finished with step : 23
139 th episode finished with step : 19
140 th episode finished with step : 20
141 th episode finished with step : 22
142 th episode finished with step : 18
143 th episode finished with step : 15
144 th episode finished with step : 15
145 th episode finished with step : 19
146 th episode finished with step : 19
147 th episode finished with step : 16
148 th episode finished with step : 19
149 th episode finished with step : 20
150 th episode finished with step : 17
151 th episode finished with step : 26
152 th episode finished with step : 18
153 th episode finished with step : 20
154 th episode finished with step : 18
155 th episode finished with step : 22
156 th episode finished with step : 19
157 th episode finished with step : 20
158 th episode finished with step : 15
159 th episode finished with step : 17
160 th episode finished with step : 20
161 th episode finished with step : 21
162 th episode finished with step : 20
163 th episode finished with step : 32
164 th episode finished with step : 19
165 th episode finished with step : 24
166 th episode finished with step : 23
167 th episode finished with step : 23
168 th episode finished with step : 51
169 th episode finished with step : 22
170 th episode finished with step : 32
171 th episode finished with step : 37
172 th episode finished with step : 30
173 th episode finished with step : 26
174 th episode finished with step : 59
175 th episode finished with step : 32
176 th episode finished with step : 28
177 th episode finished with step : 56
178 th episode finished with step : 55
179 th episode finished with step : 59
180 th episode finished with step : 70
181 th episode finished with step : 98
182 th episode finished with step : 81
183 th episode finished with step : 97
184 th episode finished with step : 80
185 th episode finished with step : 70
186 th episode finished with step : 65
187 th episode finished with step : 93
188 th episode finished with step : 36
189 th episode finished with step : 87
190 th episode finished with step : 96
191 th episode finished with step : 35
192 th episode finished with step : 38
193 th episode finished with step : 25
194 th episode finished with step : 21
195 th episode finished with step : 24
196 th episode finished with step : 29
197 th episode finished with step : 25
198 th episode finished with step : 26
199 th episode finished with step : 21
200 th episode finished with step : 23
Current 'average step' of the 'step history queue' : 50.733333333333334
201 th episode finished with step : 27
202 th episode finished with step : 19
203 th episode finished with step : 21
204 th episode finished with step : 31
205 th episode finished with step : 25
206 th episode finished with step : 23
207 th episode finished with step : 20
208 th episode finished with step : 18
209 th episode finished with step : 19
210 th episode finished with step : 21
211 th episode finished with step : 24
212 th episode finished with step : 22
213 th episode finished with step : 21
214 th episode finished with step : 17
215 th episode finished with step : 15
216 th episode finished with step : 19
217 th episode finished with step : 19
218 th episode finished with step : 17
219 th episode finished with step : 17
220 th episode finished with step : 19
221 th episode finished with step : 23
222 th episode finished with step : 20
223 th episode finished with step : 25
224 th episode finished with step : 26
225 th episode finished with step : 23
226 th episode finished with step : 22
227 th episode finished with step : 24
228 th episode finished with step : 22
229 th episode finished with step : 31
230 th episode finished with step : 28
231 th episode finished with step : 21
232 th episode finished with step : 18
233 th episode finished with step : 15
234 th episode finished with step : 16
235 th episode finished with step : 19
236 th episode finished with step : 23
237 th episode finished with step : 21
238 th episode finished with step : 23
239 th episode finished with step : 16
240 th episode finished with step : 16
241 th episode finished with step : 20
242 th episode finished with step : 17
243 th episode finished with step : 15
244 th episode finished with step : 18
245 th episode finished with step : 18
246 th episode finished with step : 15
247 th episode finished with step : 18
248 th episode finished with step : 22
249 th episode finished with step : 30
250 th episode finished with step : 26
251 th episode finished with step : 30
252 th episode finished with step : 24
253 th episode finished with step : 20
254 th episode finished with step : 25
255 th episode finished with step : 19
256 th episode finished with step : 18
257 th episode finished with step : 23
258 th episode finished with step : 23
259 th episode finished with step : 18
260 th episode finished with step : 18
261 th episode finished with step : 22
262 th episode finished with step : 22
263 th episode finished with step : 28
264 th episode finished with step : 18
265 th episode finished with step : 22
266 th episode finished with step : 21
267 th episode finished with step : 23
268 th episode finished with step : 29
269 th episode finished with step : 37
270 th episode finished with step : 22
271 th episode finished with step : 20
272 th episode finished with step : 28
273 th episode finished with step : 24
274 th episode finished with step : 40
275 th episode finished with step : 45
276 th episode finished with step : 45
277 th episode finished with step : 32
278 th episode finished with step : 17
279 th episode finished with step : 22
280 th episode finished with step : 21
281 th episode finished with step : 33
282 th episode finished with step : 31
283 th episode finished with step : 29
284 th episode finished with step : 39
285 th episode finished with step : 41
286 th episode finished with step : 25
287 th episode finished with step : 60
288 th episode finished with step : 32
289 th episode finished with step : 45
290 th episode finished with step : 48
291 th episode finished with step : 34
292 th episode finished with step : 34
293 th episode finished with step : 52
294 th episode finished with step : 41
295 th episode finished with step : 37
296 th episode finished with step : 42
297 th episode finished with step : 68
298 th episode finished with step : 61
299 th episode finished with step : 48
300 th episode finished with step : 66
Current 'average step' of the 'step history queue' : 38.666666666666664
301 th episode finished with step : 54
302 th episode finished with step : 44
303 th episode finished with step : 53
304 th episode finished with step : 38
305 th episode finished with step : 40
306 th episode finished with step : 37
307 th episode finished with step : 44
308 th episode finished with step : 38
309 th episode finished with step : 51
310 th episode finished with step : 37
311 th episode finished with step : 33
312 th episode finished with step : 44
313 th episode finished with step : 92
314 th episode finished with step : 44
315 th episode finished with step : 69
316 th episode finished with step : 31
317 th episode finished with step : 56
318 th episode finished with step : 36
319 th episode finished with step : 73
320 th episode finished with step : 47
321 th episode finished with step : 59
322 th episode finished with step : 133
323 th episode finished with step : 125
324 th episode finished with step : 43
325 th episode finished with step : 105
326 th episode finished with step : 78
327 th episode finished with step : 52
328 th episode finished with step : 43
329 th episode finished with step : 44
330 th episode finished with step : 71
331 th episode finished with step : 156
332 th episode finished with step : 76
333 th episode finished with step : 62
334 th episode finished with step : 60
335 th episode finished with step : 77
336 th episode finished with step : 94
337 th episode finished with step : 58
338 th episode finished with step : 79
339 th episode finished with step : 101
340 th episode finished with step : 126
341 th episode finished with step : 54
342 th episode finished with step : 75
343 th episode finished with step : 111
344 th episode finished with step : 84
345 th episode finished with step : 104
346 th episode finished with step : 80
347 th episode finished with step : 184
348 th episode finished with step : 200
349 th episode finished with step : 84
350 th episode finished with step : 104
351 th episode finished with step : 162
352 th episode finished with step : 84
353 th episode finished with step : 102
354 th episode finished with step : 170
355 th episode finished with step : 142
356 th episode finished with step : 200
Game cleared in 356 th episode with average step : 101.3
Average step history list :
[11.266666666666667, 50.733333333333334, 38.666666666666664]
Training finished ...

Game finished with step : 200.0
"""
