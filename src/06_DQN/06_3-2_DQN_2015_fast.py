"""
 Created by IntelliJ IDEA.
 Project: Deep-Reinforcement-Learning-for-Everybody
 ===========================================
 User: ByeongGil Jung
 Date: 2019-03-08
 Time: 오후 3:32
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
    data = np.transpose(train_batch, axes=[1, 0])

    # 0 : state_reshaped / 1 : next_state_reshaped / 2 : action / 3 : reward / 4 : done / 5 : info
    state_list = np.vstack([x[0] for x in data[0]])
    next_state_list = np.vstack([x[0] for x in data[1]])
    action_list = data[2].astype(np.int32)
    reward_list = data[3].astype(np.float32)
    done_list = data[4].astype(np.bool)

    q_state_list = main_dqn.predict(x_data=state_list)

    q_target_list = reward_list + discount_rate * np.max(target_dqn.predict(next_state_list), axis=1) * ~done_list
    q_state_list[np.arange(len(state_list)), action_list] = q_target_list
    # q_state_list[:, action_list] = q_target_list
    # -> 처음에 이렇게 했는데, 이러면 연산이 달라진다. (밑 참조)

    return main_dqn.train(x_data=state_list, y_data=q_state_list)


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
0 th episode finished with step : 64
1 th episode finished with step : 19
2 th episode finished with step : 26
3 th episode finished with step : 13
4 th episode finished with step : 13
5 th episode finished with step : 34
6 th episode finished with step : 14
7 th episode finished with step : 18
8 th episode finished with step : 12
9 th episode finished with step : 20
10 th episode finished with step : 12
11 th episode finished with step : 24
12 th episode finished with step : 36
13 th episode finished with step : 30
14 th episode finished with step : 21
15 th episode finished with step : 24
16 th episode finished with step : 36
17 th episode finished with step : 44
18 th episode finished with step : 45
19 th episode finished with step : 18
20 th episode finished with step : 17
21 th episode finished with step : 28
22 th episode finished with step : 24
23 th episode finished with step : 17
24 th episode finished with step : 15
25 th episode finished with step : 15
26 th episode finished with step : 18
27 th episode finished with step : 14
28 th episode finished with step : 12
29 th episode finished with step : 12
30 th episode finished with step : 16
31 th episode finished with step : 10
32 th episode finished with step : 10
33 th episode finished with step : 10
34 th episode finished with step : 14
35 th episode finished with step : 11
36 th episode finished with step : 11
37 th episode finished with step : 11
38 th episode finished with step : 13
39 th episode finished with step : 11
40 th episode finished with step : 11
41 th episode finished with step : 10
42 th episode finished with step : 10
43 th episode finished with step : 10
44 th episode finished with step : 10
45 th episode finished with step : 12
46 th episode finished with step : 10
47 th episode finished with step : 9
48 th episode finished with step : 12
49 th episode finished with step : 9
50 th episode finished with step : 11
51 th episode finished with step : 10
52 th episode finished with step : 9
53 th episode finished with step : 12
54 th episode finished with step : 10
55 th episode finished with step : 9
56 th episode finished with step : 10
57 th episode finished with step : 9
58 th episode finished with step : 9
59 th episode finished with step : 11
60 th episode finished with step : 9
61 th episode finished with step : 10
62 th episode finished with step : 9
63 th episode finished with step : 12
64 th episode finished with step : 10
65 th episode finished with step : 10
66 th episode finished with step : 8
67 th episode finished with step : 8
68 th episode finished with step : 11
69 th episode finished with step : 8
70 th episode finished with step : 12
71 th episode finished with step : 10
72 th episode finished with step : 13
73 th episode finished with step : 9
74 th episode finished with step : 10
75 th episode finished with step : 14
76 th episode finished with step : 10
77 th episode finished with step : 11
78 th episode finished with step : 9
79 th episode finished with step : 11
80 th episode finished with step : 10
81 th episode finished with step : 11
82 th episode finished with step : 9
83 th episode finished with step : 13
84 th episode finished with step : 10
85 th episode finished with step : 13
86 th episode finished with step : 11
87 th episode finished with step : 10
88 th episode finished with step : 10
89 th episode finished with step : 12
90 th episode finished with step : 10
91 th episode finished with step : 10
92 th episode finished with step : 9
93 th episode finished with step : 9
94 th episode finished with step : 8
95 th episode finished with step : 9
96 th episode finished with step : 9
97 th episode finished with step : 14
98 th episode finished with step : 8
99 th episode finished with step : 10
100 th episode finished with step : 9
Current 'average step' of the 'step history queue' : 10.366666666666667
101 th episode finished with step : 11
102 th episode finished with step : 11
103 th episode finished with step : 10
104 th episode finished with step : 12
105 th episode finished with step : 10
106 th episode finished with step : 13
107 th episode finished with step : 10
108 th episode finished with step : 11
109 th episode finished with step : 12
110 th episode finished with step : 11
111 th episode finished with step : 13
112 th episode finished with step : 12
113 th episode finished with step : 12
114 th episode finished with step : 9
115 th episode finished with step : 13
116 th episode finished with step : 10
117 th episode finished with step : 11
118 th episode finished with step : 14
119 th episode finished with step : 10
120 th episode finished with step : 15
121 th episode finished with step : 17
122 th episode finished with step : 13
123 th episode finished with step : 16
124 th episode finished with step : 16
125 th episode finished with step : 15
126 th episode finished with step : 19
127 th episode finished with step : 17
128 th episode finished with step : 17
129 th episode finished with step : 20
130 th episode finished with step : 16
131 th episode finished with step : 22
132 th episode finished with step : 19
133 th episode finished with step : 23
134 th episode finished with step : 25
135 th episode finished with step : 44
136 th episode finished with step : 31
137 th episode finished with step : 45
138 th episode finished with step : 25
139 th episode finished with step : 27
140 th episode finished with step : 28
141 th episode finished with step : 38
142 th episode finished with step : 31
143 th episode finished with step : 27
144 th episode finished with step : 18
145 th episode finished with step : 15
146 th episode finished with step : 18
147 th episode finished with step : 16
148 th episode finished with step : 12
149 th episode finished with step : 13
150 th episode finished with step : 13
151 th episode finished with step : 15
152 th episode finished with step : 18
153 th episode finished with step : 13
154 th episode finished with step : 19
155 th episode finished with step : 13
156 th episode finished with step : 12
157 th episode finished with step : 10
158 th episode finished with step : 11
159 th episode finished with step : 14
160 th episode finished with step : 13
161 th episode finished with step : 12
162 th episode finished with step : 11
163 th episode finished with step : 13
164 th episode finished with step : 12
165 th episode finished with step : 15
166 th episode finished with step : 11
167 th episode finished with step : 17
168 th episode finished with step : 12
169 th episode finished with step : 15
170 th episode finished with step : 12
171 th episode finished with step : 17
172 th episode finished with step : 12
173 th episode finished with step : 12
174 th episode finished with step : 12
175 th episode finished with step : 11
176 th episode finished with step : 9
177 th episode finished with step : 10
178 th episode finished with step : 11
179 th episode finished with step : 11
180 th episode finished with step : 12
181 th episode finished with step : 11
182 th episode finished with step : 12
183 th episode finished with step : 11
184 th episode finished with step : 12
185 th episode finished with step : 12
186 th episode finished with step : 12
187 th episode finished with step : 8
188 th episode finished with step : 9
189 th episode finished with step : 8
190 th episode finished with step : 9
191 th episode finished with step : 9
192 th episode finished with step : 9
193 th episode finished with step : 9
194 th episode finished with step : 10
195 th episode finished with step : 10
196 th episode finished with step : 8
197 th episode finished with step : 10
198 th episode finished with step : 9
199 th episode finished with step : 9
200 th episode finished with step : 10
Current 'average step' of the 'step history queue' : 10.466666666666667
201 th episode finished with step : 10
202 th episode finished with step : 9
203 th episode finished with step : 9
204 th episode finished with step : 9
205 th episode finished with step : 13
206 th episode finished with step : 9
207 th episode finished with step : 11
208 th episode finished with step : 10
209 th episode finished with step : 11
210 th episode finished with step : 8
211 th episode finished with step : 9
212 th episode finished with step : 10
213 th episode finished with step : 9
214 th episode finished with step : 11
215 th episode finished with step : 8
216 th episode finished with step : 10
217 th episode finished with step : 11
218 th episode finished with step : 12
219 th episode finished with step : 10
220 th episode finished with step : 13
221 th episode finished with step : 12
222 th episode finished with step : 17
223 th episode finished with step : 16
224 th episode finished with step : 19
225 th episode finished with step : 14
226 th episode finished with step : 11
227 th episode finished with step : 15
228 th episode finished with step : 15
229 th episode finished with step : 21
230 th episode finished with step : 21
231 th episode finished with step : 24
232 th episode finished with step : 19
233 th episode finished with step : 24
234 th episode finished with step : 20
235 th episode finished with step : 62
236 th episode finished with step : 40
237 th episode finished with step : 87
238 th episode finished with step : 38
239 th episode finished with step : 60
240 th episode finished with step : 52
241 th episode finished with step : 59
242 th episode finished with step : 110
243 th episode finished with step : 32
244 th episode finished with step : 73
245 th episode finished with step : 86
246 th episode finished with step : 53
247 th episode finished with step : 177
248 th episode finished with step : 48
249 th episode finished with step : 55
250 th episode finished with step : 79
251 th episode finished with step : 50
252 th episode finished with step : 52
253 th episode finished with step : 57
254 th episode finished with step : 51
255 th episode finished with step : 56
256 th episode finished with step : 39
257 th episode finished with step : 71
258 th episode finished with step : 63
259 th episode finished with step : 42
260 th episode finished with step : 46
261 th episode finished with step : 105
262 th episode finished with step : 66
263 th episode finished with step : 51
264 th episode finished with step : 47
265 th episode finished with step : 48
266 th episode finished with step : 53
267 th episode finished with step : 57
268 th episode finished with step : 55
269 th episode finished with step : 81
270 th episode finished with step : 51
271 th episode finished with step : 116
272 th episode finished with step : 43
273 th episode finished with step : 49
274 th episode finished with step : 54
275 th episode finished with step : 47
276 th episode finished with step : 46
277 th episode finished with step : 67
278 th episode finished with step : 38
279 th episode finished with step : 68
280 th episode finished with step : 74
281 th episode finished with step : 43
282 th episode finished with step : 40
283 th episode finished with step : 47
284 th episode finished with step : 53
285 th episode finished with step : 42
286 th episode finished with step : 109
287 th episode finished with step : 53
288 th episode finished with step : 52
289 th episode finished with step : 106
290 th episode finished with step : 58
291 th episode finished with step : 49
292 th episode finished with step : 67
293 th episode finished with step : 45
294 th episode finished with step : 49
295 th episode finished with step : 62
296 th episode finished with step : 63
297 th episode finished with step : 85
298 th episode finished with step : 118
299 th episode finished with step : 48
300 th episode finished with step : 54
Current 'average step' of the 'step history queue' : 61.5
301 th episode finished with step : 90
302 th episode finished with step : 47
303 th episode finished with step : 45
304 th episode finished with step : 58
305 th episode finished with step : 76
306 th episode finished with step : 105
307 th episode finished with step : 53
308 th episode finished with step : 78
309 th episode finished with step : 58
310 th episode finished with step : 60
311 th episode finished with step : 51
312 th episode finished with step : 65
313 th episode finished with step : 98
314 th episode finished with step : 67
315 th episode finished with step : 72
316 th episode finished with step : 67
317 th episode finished with step : 83
318 th episode finished with step : 76
319 th episode finished with step : 56
320 th episode finished with step : 68
321 th episode finished with step : 81
322 th episode finished with step : 93
323 th episode finished with step : 82
324 th episode finished with step : 72
325 th episode finished with step : 74
326 th episode finished with step : 200
327 th episode finished with step : 96
328 th episode finished with step : 65
329 th episode finished with step : 82
330 th episode finished with step : 143
331 th episode finished with step : 80
332 th episode finished with step : 120
333 th episode finished with step : 79
334 th episode finished with step : 104
335 th episode finished with step : 75
336 th episode finished with step : 90
337 th episode finished with step : 102
338 th episode finished with step : 138
339 th episode finished with step : 110
340 th episode finished with step : 142
341 th episode finished with step : 130
342 th episode finished with step : 93
343 th episode finished with step : 194
344 th episode finished with step : 113
345 th episode finished with step : 60
346 th episode finished with step : 169
Game cleared in 346 th episode with average step : 102.33333333333333
Average step history list :
[10.366666666666667, 10.466666666666667, 61.5]
Training finished ...

Game finished with step : 190.0
"""
