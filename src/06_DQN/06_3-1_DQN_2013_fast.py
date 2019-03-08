"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-03-07
 Time: 오후 3:27
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
    data = np.transpose(train_batch, axes=[1, 0])

    # 0 : state_reshaped / 1 : next_state_reshaped / 2 : action / 3 : reward / 4 : done / 5 : info
    state_list = np.vstack([x[0] for x in data[0]])
    next_state_list = np.vstack([x[0] for x in data[1]])
    action_list = data[2].astype(np.int32)
    reward_list = data[3].astype(np.float32)
    done_list = data[4].astype(np.bool)

    q_state_list = dqn.predict(x_test=state_list)

    q_target_list = reward_list + discount_rate * np.max(dqn.predict(x_test=next_state_list), axis=1) * ~done_list
    q_state_list[np.arange(len(state_list)), action_list] = q_target_list
    # q_state_list[:, action_list] = q_target_list
    # -> 처음에 이렇게 했는데, 이러면 연산이 달라진다. (밑 참조)

    return dqn.train(x_train=state_list, y_train=q_state_list)


"""
[ 참조 ]

> q_state_list[np.arange(len(state_list)), action_list]
> q_state_list[:, action_list]

의 결과는 서로 다르다.

1) q_state_list[np.arange(len(state_list)), action_list]
 -> 이것은 해당 ndarray 의 원하는 index 를 두 배열의 index 를 통해 참조하는 것이다.
 
 즉,
 > a[[1, 2], [0, 2]]
 를 하면
 a[1, 0] 과 a[2, 2] 를 순서대로 참조하게 된다.
 
 그리고 [a[1, 0], a[2, 2]] 을 반환한다.

2) q_state_list[:, action_list]
 -> 이것은 해당 ndarray 에서 axis=0 의 array 들의 index 를 action_list 의 index 대로 순서를 바꾸는 것이다.
 
 즉,
 > a[:, [0, 3]]
 를 하면
 a 의 axis=0 에 해당하는 모든 array 들의
 index 0 의 값과 index 3 의 값을 서로 교체시킨다.
 
 만약 바꿀 array 의 length 를 초과하는 범위의 index 를 넣었다면,
 초과된 index 는 broadcast 되어 반환된다.

=========================================

ex)
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
c = np.array([100, 200, 300, 400, 500])

n = np.array(list(zip(a, b, c)))

print(n)
print(n[:, [2, 0, 1]])
print(n[:, [0, 1, 0, 2]])  # broadcast 발생
print(n[[0, 1, 3, 4], [0, 1, 0, 2]])

>>
[[  1  10 100]
 [  2  20 200]
 [  3  30 300]
 [  4  40 400]
 [  5  50 500]]
 
[[100   1  10]
 [200   2  20]
 [300   3  30]
 [400   4  40]
 [500   5  50]]
 
[[  1  10   1 100]
 [  2  20   2 200]
 [  3  30   3 300]
 [  4  40   4 400]
 [  5  50   5 500]]
 
[  1  20   4 500]
"""


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
0 th episode finished with steps : 26
1 th episode finished with steps : 14
2 th episode finished with steps : 11
3 th episode finished with steps : 33
4 th episode finished with steps : 19
5 th episode finished with steps : 19
6 th episode finished with steps : 16
7 th episode finished with steps : 10
8 th episode finished with steps : 21
9 th episode finished with steps : 14
10 th episode finished with steps : 19
11 th episode finished with steps : 13
12 th episode finished with steps : 13
13 th episode finished with steps : 9
14 th episode finished with steps : 11
15 th episode finished with steps : 9
16 th episode finished with steps : 12
17 th episode finished with steps : 12
18 th episode finished with steps : 14
19 th episode finished with steps : 20
20 th episode finished with steps : 10
21 th episode finished with steps : 13
22 th episode finished with steps : 10
23 th episode finished with steps : 12
24 th episode finished with steps : 12
25 th episode finished with steps : 13
26 th episode finished with steps : 10
27 th episode finished with steps : 9
28 th episode finished with steps : 11
29 th episode finished with steps : 10
30 th episode finished with steps : 13
31 th episode finished with steps : 11
32 th episode finished with steps : 8
33 th episode finished with steps : 12
34 th episode finished with steps : 9
35 th episode finished with steps : 8
36 th episode finished with steps : 11
37 th episode finished with steps : 9
38 th episode finished with steps : 9
39 th episode finished with steps : 10
40 th episode finished with steps : 10
41 th episode finished with steps : 11
42 th episode finished with steps : 10
43 th episode finished with steps : 8
44 th episode finished with steps : 9
45 th episode finished with steps : 12
46 th episode finished with steps : 9
47 th episode finished with steps : 12
48 th episode finished with steps : 13
49 th episode finished with steps : 10
50 th episode finished with steps : 9
51 th episode finished with steps : 10
52 th episode finished with steps : 14
53 th episode finished with steps : 8
54 th episode finished with steps : 13
55 th episode finished with steps : 12
56 th episode finished with steps : 9
57 th episode finished with steps : 11
58 th episode finished with steps : 9
59 th episode finished with steps : 12
60 th episode finished with steps : 10
61 th episode finished with steps : 12
62 th episode finished with steps : 9
63 th episode finished with steps : 12
64 th episode finished with steps : 9
65 th episode finished with steps : 9
66 th episode finished with steps : 10
67 th episode finished with steps : 12
68 th episode finished with steps : 10
69 th episode finished with steps : 9
70 th episode finished with steps : 10
71 th episode finished with steps : 11
72 th episode finished with steps : 10
73 th episode finished with steps : 9
74 th episode finished with steps : 11
75 th episode finished with steps : 10
76 th episode finished with steps : 11
77 th episode finished with steps : 10
78 th episode finished with steps : 10
79 th episode finished with steps : 10
80 th episode finished with steps : 14
81 th episode finished with steps : 12
82 th episode finished with steps : 11
83 th episode finished with steps : 12
84 th episode finished with steps : 9
85 th episode finished with steps : 11
86 th episode finished with steps : 9
87 th episode finished with steps : 11
88 th episode finished with steps : 9
89 th episode finished with steps : 11
90 th episode finished with steps : 10
91 th episode finished with steps : 9
92 th episode finished with steps : 8
93 th episode finished with steps : 10
94 th episode finished with steps : 10
95 th episode finished with steps : 9
96 th episode finished with steps : 11
97 th episode finished with steps : 10
98 th episode finished with steps : 11
99 th episode finished with steps : 10
100 th episode finished with steps : 12
Current 'average step' of the 'step history queue' : 10.366666666666667
101 th episode finished with steps : 13
102 th episode finished with steps : 10
103 th episode finished with steps : 10
104 th episode finished with steps : 10
105 th episode finished with steps : 10
106 th episode finished with steps : 11
107 th episode finished with steps : 10
108 th episode finished with steps : 12
109 th episode finished with steps : 13
110 th episode finished with steps : 10
111 th episode finished with steps : 12
112 th episode finished with steps : 12
113 th episode finished with steps : 10
114 th episode finished with steps : 12
115 th episode finished with steps : 10
116 th episode finished with steps : 14
117 th episode finished with steps : 12
118 th episode finished with steps : 13
119 th episode finished with steps : 9
120 th episode finished with steps : 11
121 th episode finished with steps : 14
122 th episode finished with steps : 15
123 th episode finished with steps : 11
124 th episode finished with steps : 11
125 th episode finished with steps : 12
126 th episode finished with steps : 10
127 th episode finished with steps : 13
128 th episode finished with steps : 11
129 th episode finished with steps : 10
130 th episode finished with steps : 12
131 th episode finished with steps : 17
132 th episode finished with steps : 18
133 th episode finished with steps : 16
134 th episode finished with steps : 18
135 th episode finished with steps : 12
136 th episode finished with steps : 12
137 th episode finished with steps : 19
138 th episode finished with steps : 21
139 th episode finished with steps : 15
140 th episode finished with steps : 15
141 th episode finished with steps : 19
142 th episode finished with steps : 17
143 th episode finished with steps : 28
144 th episode finished with steps : 28
145 th episode finished with steps : 23
146 th episode finished with steps : 9
147 th episode finished with steps : 9
148 th episode finished with steps : 9
149 th episode finished with steps : 10
150 th episode finished with steps : 9
151 th episode finished with steps : 9
152 th episode finished with steps : 8
153 th episode finished with steps : 9
154 th episode finished with steps : 11
155 th episode finished with steps : 18
156 th episode finished with steps : 9
157 th episode finished with steps : 26
158 th episode finished with steps : 29
159 th episode finished with steps : 24
160 th episode finished with steps : 24
161 th episode finished with steps : 28
162 th episode finished with steps : 32
163 th episode finished with steps : 27
164 th episode finished with steps : 25
165 th episode finished with steps : 10
166 th episode finished with steps : 28
167 th episode finished with steps : 28
168 th episode finished with steps : 98
169 th episode finished with steps : 38
170 th episode finished with steps : 39
171 th episode finished with steps : 71
172 th episode finished with steps : 85
173 th episode finished with steps : 88
174 th episode finished with steps : 68
175 th episode finished with steps : 32
176 th episode finished with steps : 42
177 th episode finished with steps : 40
178 th episode finished with steps : 41
179 th episode finished with steps : 19
180 th episode finished with steps : 20
181 th episode finished with steps : 44
182 th episode finished with steps : 65
183 th episode finished with steps : 45
184 th episode finished with steps : 22
185 th episode finished with steps : 23
186 th episode finished with steps : 27
187 th episode finished with steps : 39
188 th episode finished with steps : 32
189 th episode finished with steps : 32
190 th episode finished with steps : 22
191 th episode finished with steps : 24
192 th episode finished with steps : 23
193 th episode finished with steps : 29
194 th episode finished with steps : 23
195 th episode finished with steps : 24
196 th episode finished with steps : 31
197 th episode finished with steps : 36
198 th episode finished with steps : 20
199 th episode finished with steps : 28
200 th episode finished with steps : 20
Current 'average step' of the 'step history queue' : 37.166666666666664
201 th episode finished with steps : 19
202 th episode finished with steps : 29
203 th episode finished with steps : 25
204 th episode finished with steps : 20
205 th episode finished with steps : 19
206 th episode finished with steps : 18
207 th episode finished with steps : 28
208 th episode finished with steps : 17
209 th episode finished with steps : 38
210 th episode finished with steps : 20
211 th episode finished with steps : 26
212 th episode finished with steps : 16
213 th episode finished with steps : 20
214 th episode finished with steps : 33
215 th episode finished with steps : 26
216 th episode finished with steps : 26
217 th episode finished with steps : 28
218 th episode finished with steps : 28
219 th episode finished with steps : 24
220 th episode finished with steps : 24
221 th episode finished with steps : 30
222 th episode finished with steps : 26
223 th episode finished with steps : 23
224 th episode finished with steps : 28
225 th episode finished with steps : 20
226 th episode finished with steps : 19
227 th episode finished with steps : 26
228 th episode finished with steps : 28
229 th episode finished with steps : 41
230 th episode finished with steps : 28
231 th episode finished with steps : 28
232 th episode finished with steps : 23
233 th episode finished with steps : 21
234 th episode finished with steps : 30
235 th episode finished with steps : 25
236 th episode finished with steps : 20
237 th episode finished with steps : 24
238 th episode finished with steps : 20
239 th episode finished with steps : 33
240 th episode finished with steps : 36
241 th episode finished with steps : 31
242 th episode finished with steps : 27
243 th episode finished with steps : 24
244 th episode finished with steps : 40
245 th episode finished with steps : 43
246 th episode finished with steps : 22
247 th episode finished with steps : 24
248 th episode finished with steps : 20
249 th episode finished with steps : 43
250 th episode finished with steps : 47
251 th episode finished with steps : 30
252 th episode finished with steps : 37
253 th episode finished with steps : 22
254 th episode finished with steps : 19
255 th episode finished with steps : 22
256 th episode finished with steps : 28
257 th episode finished with steps : 36
258 th episode finished with steps : 27
259 th episode finished with steps : 30
260 th episode finished with steps : 31
261 th episode finished with steps : 24
262 th episode finished with steps : 22
263 th episode finished with steps : 31
264 th episode finished with steps : 29
265 th episode finished with steps : 41
266 th episode finished with steps : 30
267 th episode finished with steps : 30
268 th episode finished with steps : 35
269 th episode finished with steps : 32
270 th episode finished with steps : 28
271 th episode finished with steps : 31
272 th episode finished with steps : 32
273 th episode finished with steps : 34
274 th episode finished with steps : 28
275 th episode finished with steps : 20
276 th episode finished with steps : 25
277 th episode finished with steps : 28
278 th episode finished with steps : 32
279 th episode finished with steps : 28
280 th episode finished with steps : 39
281 th episode finished with steps : 33
282 th episode finished with steps : 24
283 th episode finished with steps : 28
284 th episode finished with steps : 23
285 th episode finished with steps : 21
286 th episode finished with steps : 23
287 th episode finished with steps : 33
288 th episode finished with steps : 23
289 th episode finished with steps : 30
290 th episode finished with steps : 33
291 th episode finished with steps : 33
292 th episode finished with steps : 36
293 th episode finished with steps : 29
294 th episode finished with steps : 39
295 th episode finished with steps : 25
296 th episode finished with steps : 34
297 th episode finished with steps : 28
298 th episode finished with steps : 23
299 th episode finished with steps : 26
300 th episode finished with steps : 35
Current 'average step' of the 'step history queue' : 29.2
301 th episode finished with steps : 23
302 th episode finished with steps : 31
303 th episode finished with steps : 31
304 th episode finished with steps : 58
305 th episode finished with steps : 37
306 th episode finished with steps : 28
307 th episode finished with steps : 26
308 th episode finished with steps : 41
309 th episode finished with steps : 49
310 th episode finished with steps : 48
311 th episode finished with steps : 36
312 th episode finished with steps : 32
313 th episode finished with steps : 32
314 th episode finished with steps : 45
315 th episode finished with steps : 32
316 th episode finished with steps : 29
317 th episode finished with steps : 35
318 th episode finished with steps : 30
319 th episode finished with steps : 35
320 th episode finished with steps : 34
321 th episode finished with steps : 30
322 th episode finished with steps : 29
323 th episode finished with steps : 30
324 th episode finished with steps : 24
325 th episode finished with steps : 30
326 th episode finished with steps : 39
327 th episode finished with steps : 27
328 th episode finished with steps : 42
329 th episode finished with steps : 27
330 th episode finished with steps : 37
331 th episode finished with steps : 28
332 th episode finished with steps : 42
333 th episode finished with steps : 45
334 th episode finished with steps : 39
335 th episode finished with steps : 41
336 th episode finished with steps : 29
337 th episode finished with steps : 41
338 th episode finished with steps : 39
339 th episode finished with steps : 54
340 th episode finished with steps : 42
341 th episode finished with steps : 38
342 th episode finished with steps : 44
343 th episode finished with steps : 29
344 th episode finished with steps : 38
345 th episode finished with steps : 33
346 th episode finished with steps : 37
347 th episode finished with steps : 34
348 th episode finished with steps : 39
349 th episode finished with steps : 33
350 th episode finished with steps : 49
351 th episode finished with steps : 54
352 th episode finished with steps : 42
353 th episode finished with steps : 29
354 th episode finished with steps : 46
355 th episode finished with steps : 35
356 th episode finished with steps : 47
357 th episode finished with steps : 38
358 th episode finished with steps : 31
359 th episode finished with steps : 46
360 th episode finished with steps : 56
361 th episode finished with steps : 48
362 th episode finished with steps : 33
363 th episode finished with steps : 33
364 th episode finished with steps : 59
365 th episode finished with steps : 34
366 th episode finished with steps : 40
367 th episode finished with steps : 42
368 th episode finished with steps : 25
369 th episode finished with steps : 31
370 th episode finished with steps : 27
371 th episode finished with steps : 28
372 th episode finished with steps : 30
373 th episode finished with steps : 32
374 th episode finished with steps : 44
375 th episode finished with steps : 52
376 th episode finished with steps : 31
377 th episode finished with steps : 43
378 th episode finished with steps : 31
379 th episode finished with steps : 33
380 th episode finished with steps : 27
381 th episode finished with steps : 40
382 th episode finished with steps : 32
383 th episode finished with steps : 44
384 th episode finished with steps : 37
385 th episode finished with steps : 42
386 th episode finished with steps : 42
387 th episode finished with steps : 42
388 th episode finished with steps : 41
389 th episode finished with steps : 41
390 th episode finished with steps : 44
391 th episode finished with steps : 46
392 th episode finished with steps : 32
393 th episode finished with steps : 39
394 th episode finished with steps : 39
395 th episode finished with steps : 59
396 th episode finished with steps : 42
397 th episode finished with steps : 49
398 th episode finished with steps : 37
399 th episode finished with steps : 47
400 th episode finished with steps : 30
Current 'average step' of the 'step history queue' : 39.2
401 th episode finished with steps : 49
402 th episode finished with steps : 35
403 th episode finished with steps : 51
404 th episode finished with steps : 47
405 th episode finished with steps : 34
406 th episode finished with steps : 32
407 th episode finished with steps : 32
408 th episode finished with steps : 36
409 th episode finished with steps : 53
410 th episode finished with steps : 40
411 th episode finished with steps : 51
412 th episode finished with steps : 43
413 th episode finished with steps : 83
414 th episode finished with steps : 75
415 th episode finished with steps : 86
416 th episode finished with steps : 85
417 th episode finished with steps : 80
418 th episode finished with steps : 91
419 th episode finished with steps : 79
420 th episode finished with steps : 81
421 th episode finished with steps : 94
422 th episode finished with steps : 73
423 th episode finished with steps : 108
424 th episode finished with steps : 94
425 th episode finished with steps : 72
426 th episode finished with steps : 49
427 th episode finished with steps : 105
428 th episode finished with steps : 94
429 th episode finished with steps : 79
430 th episode finished with steps : 114
431 th episode finished with steps : 92
432 th episode finished with steps : 82
433 th episode finished with steps : 104
434 th episode finished with steps : 74
435 th episode finished with steps : 99
436 th episode finished with steps : 101
437 th episode finished with steps : 90
438 th episode finished with steps : 92
439 th episode finished with steps : 89
440 th episode finished with steps : 93
441 th episode finished with steps : 104
442 th episode finished with steps : 76
443 th episode finished with steps : 83
444 th episode finished with steps : 102
445 th episode finished with steps : 87
446 th episode finished with steps : 86
447 th episode finished with steps : 80
448 th episode finished with steps : 79
449 th episode finished with steps : 83
450 th episode finished with steps : 87
451 th episode finished with steps : 87
452 th episode finished with steps : 115
453 th episode finished with steps : 91
454 th episode finished with steps : 79
455 th episode finished with steps : 89
456 th episode finished with steps : 83
457 th episode finished with steps : 71
458 th episode finished with steps : 86
459 th episode finished with steps : 86
460 th episode finished with steps : 75
461 th episode finished with steps : 109
462 th episode finished with steps : 81
463 th episode finished with steps : 97
464 th episode finished with steps : 99
465 th episode finished with steps : 102
466 th episode finished with steps : 80
467 th episode finished with steps : 84
468 th episode finished with steps : 77
469 th episode finished with steps : 83
470 th episode finished with steps : 94
471 th episode finished with steps : 78
472 th episode finished with steps : 94
473 th episode finished with steps : 97
474 th episode finished with steps : 81
475 th episode finished with steps : 103
476 th episode finished with steps : 104
477 th episode finished with steps : 83
478 th episode finished with steps : 92
479 th episode finished with steps : 119
480 th episode finished with steps : 78
481 th episode finished with steps : 73
482 th episode finished with steps : 98
483 th episode finished with steps : 86
484 th episode finished with steps : 118
485 th episode finished with steps : 123
486 th episode finished with steps : 97
487 th episode finished with steps : 116
488 th episode finished with steps : 92
489 th episode finished with steps : 145
490 th episode finished with steps : 110
491 th episode finished with steps : 94
492 th episode finished with steps : 90
493 th episode finished with steps : 78
494 th episode finished with steps : 96
495 th episode finished with steps : 84
496 th episode finished with steps : 92
497 th episode finished with steps : 168
498 th episode finished with steps : 100
499 th episode finished with steps : 87
500 th episode finished with steps : 106
Current 'average step' of the 'step history queue' : 99.4
501 th episode finished with steps : 98
Game cleared in 501 th episode with average step : 100.06666666666666
Average step history list :
[10.366666666666667, 37.166666666666664, 29.2, 39.2, 99.4]
Training finished ...

Game finished with step : 79.0
"""
