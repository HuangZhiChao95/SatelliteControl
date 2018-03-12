import tensorflow as tf
import numpy as np
from multiprocessing import Process, Semaphore, Queue
from env.SatelliteEnv import SatelliteEnv
from env.EnvBlock import envblock


class Model:
    def __init__(self, unit=[128, 256, 512, 512], state_dim=7, action_dim=3, init_std=1e-2, act=tf.nn.relu,
                 process_num=4, batch_size=16, buffer_size=100):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim + action_dim], name="input")
        self.output = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="action")
        self.lr_rate = tf.placeholder(dtype=tf.float32, name="lr_rate")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")

        self.batch_size = batch_size
        self.process_num = process_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size * self.process_num * self.batch_size
        self.states = np.zeros([self.buffer_size, self.state_dim], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, self.action_dim], dtype=np.float32)
        self.next_states = np.zeros([self.buffer_size, self.state_dim], dtype=np.float32)
        self.head = 0

        unit = [state_dim + action_dim] + unit + [state_dim]
        network = self.input

        for i in range(0, len(unit) - 1):
            if i == len(unit) - 2:
                act = tf.identity
                batch_norm = False
            else:
                batch_norm = True
            with tf.name_scope("dense{0}".format(i + 1)):
                network = self._denselayer(network, unit[i], unit[i + 1], init_std, act, batch_norm)

        self.next_state_model = network
        self.loss = tf.reduce_mean(tf.abs(network - self.output))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss)
        self._init_env()

    def _init_env(self, tspan=0.5):
        penv = {
            "tspan": tspan,
            "theta": np.matmul(2 * np.random.rand(5000, 3) - 1, np.diag([1, 1, 1])),
            "wb": np.matmul(2 * np.random.rand(5000, 3) - 1, np.diag([0.01, 0.01, 0.01]))
        }
        p_testenv = {
            "tspan": tspan,
            "theta": np.array([[0.9, 0.9, 0.4]]),
            "wb": np.ones((1, 3)) * 0.01 * np.pi / 180
        }
        self.test_env = SatelliteEnv(p_testenv)
        self.processes = list()
        self.queues = list()
        self.lockenvs = list()
        self.lockmains = list()

        for i in range(self.process_num):
            lockenv = Semaphore(0)
            lockmain = Semaphore(0)
            q = Queue()
            process = Process(target=envblock, args=(self.batch_size, penv, lockenv, lockmain, q, 7))
            self.processes.append(process)
            self.lockenvs.append(lockenv)
            self.lockmains.append(lockmain)
            self.queues.append(q)

            process.start()

        self._store(self.buffer_size // (self.process_num * self.batch_size))

    def _batch_norm(self, input, dim):

        batch_mean, batch_var = tf.nn.moments(input, [0])
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        beta = tf.Variable(tf.zeros(shape=[dim]))
        gamma = tf.Variable(tf.ones(shape=[dim]))
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-4)

    def _denselayer(self, input, n_in, n_out, init_std, activation=tf.nn.relu, batch_norm=True):

        w = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=init_std), name="weight")
        b = tf.Variable(tf.zeros([n_out]), name="bias")
        relu = activation(tf.nn.bias_add(tf.matmul(input, w), b), name="relu")
        if batch_norm:
            output = self._batch_norm(relu, n_out)
        else:
            output = relu
        tf.summary.histogram(values=w, name="weight")
        tf.summary.histogram(values=b, name="bias")
        tf.summary.histogram(values=relu, name="relu")
        tf.summary.histogram(values=output, name="output")
        return output

    def predict(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        merge_op = tf.summary.merge_all()
        input = np.concatenate((state[np.newaxis, :], action[np.newaxis, :]), axis=1)
        print(input)
        next_state,summary_str = sess.run([self.next_state_model,merge_op], feed_dict={self.input: input, self.phase_train: True})
        print(next_state)

        return next_state.squeeze(),summary_str

    def update(self, state, action, next_state, lr_rate, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {
            self.input: np.concatenate((state, action), axis=1),
            self.output: next_state,
            self.lr_rate: lr_rate,
            self.phase_train: True
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def _store(self, iteration):
        states = np.zeros([self.batch_size * self.process_num, self.state_dim], dtype=np.float32)
        for j in range(self.process_num):
            self.queues[j].put("reset")
            self.lockenvs[j].release()

        for j in range(self.process_num):
            self.lockmains[j].acquire()

        for j in range(self.process_num):
            states[self.batch_size * j:(j + 1) * self.batch_size, :] = self.queues[j].get()

        for i in range(iteration):
            actions = np.random.randn(self.process_num * self.batch_size, self.action_dim) * 5e-2
            self.actions[self.head:self.head + self.batch_size * self.process_num, :] = actions
            self.states[self.head:self.head + self.batch_size * self.process_num, :] = states

            for j in range(self.process_num):
                self.queues[j].put(actions[self.batch_size * j:(j + 1) * self.batch_size, :])
                self.lockenvs[j].release()

            for j in range(self.process_num):
                self.lockmains[j].acquire()

            for j in range(self.process_num):
                state_block, _, __ = self.queues[j].get()
                self.next_states[self.head:self.head + self.batch_size, :] = state_block
                self.head = (self.head + self.batch_size) % self.buffer_size

    def run(self, lr_rate, iteration):
        summary_writer = tf.summary.FileWriter("./log")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                if i % 10 == 0:
                    self._store(10)
                if i % 100 == 0 and i!=0:
                    state_env = self.test_env.reset()
                    state_model = state_env.copy()
                    for j in range(2000):
                        action = np.random.randn(3) * 5e-2
                        state_env, _, __, ___ = self.test_env.step(action)
                        state_model, summary_str = self.predict(state_model, action)
                        print("iteration={0} step={1} l1_error={2}".format(i, j, np.sum(np.abs(state_model - state_env))))
                        summary_writer.add_summary(summary_str, j)
                index = np.random.rand(64).astype(np.int32)
                loss = self.update(self.states[index], self.actions[index], self.next_states[index], lr_rate)
                print("iteration={0} loss={1}".format(i,loss))
                if i % 10000 == 0 and i != 0:
                    lr_rate = lr_rate / 2
