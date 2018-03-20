import tensorflow as tf
import numpy as np
from multiprocessing import Process, Semaphore, Queue
from env.SatelliteEnv import SatelliteEnv
from env.EnvBlock import envblock


class Model:
    def __init__(self, unit=[128,128,128,256,256], state_dim=7, action_dim=3, init_std=1e-2, act=tf.nn.relu,
                 process_num=4, batch_size=16, buffer_size=1000, tspan=0.5):
        self.state_dim = state_dim
        state_dim = 6
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim + action_dim], name="input")
        self.output = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="action")
        self.lr_rate = tf.placeholder(dtype=tf.float32, name="lr_rate")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")
        self.w_weight = tf.placeholder(dtype=tf.float32, name="w_weight")
        self.tspan = tspan

        self.batch_size = batch_size
        self.process_num = process_num
        self.action_dim = action_dim
        self.buffer_size = buffer_size * self.process_num * self.batch_size
        self.states = np.zeros([self.buffer_size, self.state_dim], dtype=np.float32)
        self.actions = np.zeros([self.buffer_size, self.action_dim], dtype=np.float32)
        self.next_states = np.zeros([self.buffer_size, self.state_dim], dtype=np.float32)
        self.head = 0
        self.env_iteration = 0

        unit = [state_dim + action_dim] + unit + [state_dim]
        with tf.name_scope("input"):
            network = self._batch_norm(self.input, state_dim+action_dim)

        for i in range(0, len(unit) - 1):
            if i == len(unit) - 2:
                act = tf.identity
                batch_norm = False
            else:
                batch_norm = True
            with tf.name_scope("dense{0}".format(i + 1)):
                network = self._denselayer(network, unit[i], unit[i + 1], init_std, act, batch_norm)
        #network = network/10
        q = tf.slice(self.input,[0,0],[-1,3])
        w = tf.slice(self.input,[0,3],[-1,3])
        a = tf.slice(self.input,[0,6],[-1,3])
        q_next = self._linear_op(w, 50, 1, "kq1") + q#self._linear_op(a, 0.05, 2, "kq2") + q
        w_next = self._linear_op(a, 50, 1, "kw1") + w#self._linear_op(a, 0.05, 2, "kw2") + w
        next_state_linear = tf.concat([q,w],axis=1)
        self.next_state_model = network + next_state_linear
        #print(self.output.shape)
        #print(self.next_state_model.shape)
        diff = tf.abs(self.next_state_model - self.output)
        diff_q = tf.reduce_mean(tf.slice(diff, [0,0],[-1,3]))
        diff_w = tf.reduce_mean(tf.slice(diff, [0,0],[-1,3]))*self.w_weight
        self.loss = diff_q+diff_w
        tf.summary.scalar(tensor=self.loss, name="loss")
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss)
        self._init_env()
    
    def _linear_op(self, input, init, p, name):
        k = tf.Variable(init*self.tspan, name=name)
        tf.summary.scalar(tensor=k, name=name)
        tmp = input
        for i in range(p-1):
            tmp = tf.multiply(tmp,input)
        return k*tmp

    def _init_env(self):
        penv = {
            "tspan": self.tspan,
            "theta": np.diag([1, 1, 1]),
            "wb": np.diag([0.01, 0.01, 0.01])
        }
        p_testenv = {
            "tspan": self.tspan,
            "theta": np.array([[0.1, 0.1, 0.1]]),
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
        return tf.identity(input)
        batch_mean, batch_var = tf.nn.moments(input, [0])
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        beta = tf.Variable(tf.zeros(shape=[dim]))
        gamma = tf.Variable(tf.ones(shape=[dim]))
        tf.summary.histogram(values=ema.average(batch_mean), name="batch_mean")
        tf.summary.histogram(values=ema.average(batch_var), name="batch_var")
        tf.summary.histogram(values=beta, name="beta")
        tf.summary.histogram(values=gamma, name="gamma")
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

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
        input = np.concatenate((state[np.newaxis, :], action[np.newaxis, :]), axis=1)
        next_state = sess.run(self.next_state_model, feed_dict={self.input: input, self.phase_train: False})
        return next_state.squeeze()

    def update(self, state, action, next_state, lr_rate, sess=None, summary=False):
        sess = sess or tf.get_default_session()
        merge_op = tf.summary.merge_all()
        feed_dict = {
            self.input: np.concatenate((state, action), axis=1),
            self.output: next_state,
            self.lr_rate: lr_rate,
            self.phase_train: True,
            self.w_weight:20
        }
        #print((next_state[:,0:3]-state[:,0:3])/state[:,3:6])
        #print((next_state[:,3:6]-state[:,3:6])/action)
        if summary:
            _, loss,summary_str,debug = sess.run([self.train_op, self.loss, merge_op, self.next_state_model], feed_dict=feed_dict)
            #print(debug)
            return loss, summary_str
        else:
            sess.run(self.train_op, feed_dict=feed_dict)

    def _store(self, iteration):
        states = np.zeros([self.batch_size * self.process_num, self.state_dim], dtype=np.float32)
        for i in range(iteration):
            if self.env_iteration==0:
                for j in range(self.process_num):
                    self.queues[j].put("reset")
                    self.lockenvs[j].release()

                for j in range(self.process_num):
                    self.lockmains[j].acquire()

                for j in range(self.process_num):
                    states[self.batch_size * j:(j + 1) * self.batch_size, :] = self.queues[j].get()
                
            
            actions = -0.05*states[:,1:4]-0.05*states[:,4:7]#+np.random.randn(self.process_num * self.batch_size, self.action_dim) * 1e-3
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
                states[j*self.batch_size:(j+1)*self.batch_size,:]=state_block
                
            self.env_iteration = (self.env_iteration+1)%1000

    def run(self, lr_rate, iteration):
        summary_writer = tf.summary.FileWriter("./log")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iteration):
                if i % 10 == 0:
                    self._store(10)
                if i % 1000 == 0 and i!=0:
                    state_env = self.test_env.reset()
                    for j in range(2000):
                        action = -0.05*state_env[1:4]-0.05*state_env[4:7] #np.random.randn(3) * 1e-2
                        state_model = self.predict(state_env[1:], action)
                        if j % 100 == 0:
                            print("iteration={0} step={1} l1_error={2}".format(i, j, np.sum(np.abs((state_model - state_env[1:])/state_env[1:]))))
                            print(state_model-state_env[1:])
                            print(state_env[1:])
                        state_env, _, __, ___ = self.test_env.step(action)
                index = (np.random.rand(32)*self.buffer_size).astype(np.int32)
                if i % 100 == 0:
                    #print(self.states)
                    #print((self.states[index,1:] - self.next_states[index,1:])/self.states[index,1:])
                    loss,summary_str = self.update(self.states[index,1:], self.actions[index], self.next_states[index,1:], lr_rate, summary=True)
                    summary_writer.add_summary(summary_str, i)
                    print("iteration={0} loss={1}".format(i,loss))
                else:
                    loss = self.update(self.states[index,1:], self.actions[index], self.next_states[index,1:], lr_rate)
                    
                if i % 4000 == 0 and i != 0:
                    lr_rate = lr_rate / 2
