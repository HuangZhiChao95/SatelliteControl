import numpy as np
import tensorflow as tf
from LQR.Model import Model
from multiprocessing import Process, Semaphore, Queue
from env.SatelliteEnv import SatelliteEnv
from env.EnvBlock import envblock
import os


class LQR:
    def __init__(self, init_std=1e-1, tpsan=0.5, batch_size=16, process_num=4, state_dim=6, action_dim=3, T=200,
                 savename="lqr"):
        self.model = Model(init_std=init_std)
        self.tspan = tpsan
        self.T = int(T / self.tspan)
        self.batch_size = batch_size
        self.process_num = process_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._init_env()
        self.xhat = np.zeros((self.T, self.state_dim))
        self.uhat = np.zeros((self.T, self.action_dim))
        self.K_list = np.zeros((self.T, self.action_dim, self.state_dim))
        self.k_list = np.zeros((self.T, self.action_dim))
        self.V_list = np.zeros((self.T, self.state_dim, self.state_dim))
        self.v_list = np.zeros((self.T, self.state_dim))
        self.Q_list = np.zeros((self.T, self.state_dim + self.action_dim, self.state_dim + self.action_dim))
        self.q_list = np.zeros((self.T, self.state_dim + self.action_dim))
        self.env_iteration = 0
        self.savename = savename
        self._init_env()

    def _init_env(self):
        penv = {
            "tspan": self.tspan,
            "theta": np.diag([0.5, 0.5, 0.5]),
            "wb": np.diag([0.01, 0.01, 0.01])
        }
        p_testenv = {
            "tspan": self.tspan,
            "theta": np.array([[0.5, 0.4, 0.3]]),
            "wb": np.ones((1, 3)) * 0.01
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
            process = Process(target=envblock, args=(self.batch_size, penv, lockenv, lockmain, q, 6))
            self.processes.append(process)
            self.lockenvs.append(lockenv)
            self.lockmains.append(lockmain)
            self.queues.append(q)

            process.start()

    def collect_sample(self):
        states = np.zeros([self.batch_size * self.process_num * self.T, self.state_dim], dtype=np.float32)
        actions = np.zeros([self.batch_size * self.process_num * self.T, self.action_dim], dtype=np.float32)
        next_states = np.zeros([self.batch_size * self.process_num * self.T, self.state_dim], dtype=np.float32)
        action = np.zeros([self.batch_size * self.process_num, self.action_dim], dtype=np.float32)
        state = np.zeros([self.batch_size * self.process_num, self.state_dim], dtype=np.float32)
        head = 0
        for j in range(self.process_num):
            self.queues[j].put("reset")
            self.lockenvs[j].release()

        for j in range(self.process_num):
            self.lockmains[j].acquire()

        for j in range(self.process_num):
            state[self.batch_size * j:(j + 1) * self.batch_size, :] = self.queues[j].get()

        for i in range(self.T):
            #print(i)
            for j in range(self.batch_size * self.process_num):
                action[j] = np.matmul(self.K_list[i], state[j] - self.xhat[i]) + self.k_list[i] + self.uhat[i]

            actions[head:head + self.batch_size * self.process_num, :] = action
            states[head:head + self.batch_size * self.process_num, :] = state

            for j in range(self.process_num):
                self.queues[j].put(action[self.batch_size * j:(j + 1) * self.batch_size, :])
                self.lockenvs[j].release()

            for j in range(self.process_num):
                self.lockmains[j].acquire()

            for j in range(self.process_num):
                state_block, _, __ = self.queues[j].get()
                #for k in range(self.batch_size):
                #    print("{0} {1}".format(k,state_block[k]))
                #if i==0:
                #    print("action")
                #    print((state_block[:,3:6]-state[j * self.batch_size:(j + 1) * self.batch_size, 3:6])/action[j * self.batch_size:(j + 1) * self.batch_size, :])
                # print("w_v")
                # print((state_block[:,0:3]-state[j * self.batch_size:(j + 1) * self.batch_size, 0:3])/state[j * self.batch_size:(j + 1) * self.batch_size, 3:6])
                next_states[head:head + self.batch_size, :] = state_block
                head = head + self.batch_size
                state[j * self.batch_size:(j + 1) * self.batch_size, :] = state_block

        self.model.store(states, actions, next_states)

    def store_record(self, iteration):
        state_env = self.test_env.reset()
        state_list = []
        action_list = []
        for i in range(self.T):
            action = np.matmul(self.K_list[i], state_env - self.xhat[i]) + self.k_list[i] + self.uhat[i]
            action_list.append(action)
            state_list.append(state_env)
            state_model = self.model.predict(state_env, action)
            state_env, _, __, ___ = self.test_env.step(action)
            if i % 50 == 0:
                print("iteration={0} step={1} l1_error={2}".format(iteration, i, np.sum(
                    np.abs((state_model - state_env) / state_env))))
                print("state diff")
                print(state_model - state_env)
                print("state_env")
                print(state_env)

        result = {
            "state": np.array(state_list),
            "action": np.array(action_list),
        }

        if not os.path.exists("record"):
            os.mkdir("record")
        print("./record/{1}_{0}".format(iteration, self.savename))
        np.save("./record/{1}_{0}".format(iteration, self.savename), np.array(result))

    def _backward(self):
        V = np.zeros((6, 6))
        v = np.zeros(6)
        print("backward")
        for i in reversed(range(self.T)):
            F = self.model.getJocobi(self.xhat[i], self.uhat[i])
            Q = np.eye(9) + np.matmul(np.matmul(np.transpose(F), V), F)
            q = np.matmul(np.transpose(F), v)
            K = -np.matmul(np.linalg.inv(Q[6:, 6:]), Q[6:, 0:6])
            if i==0:
                print("F_sample T=0")
                print(F)
                print("K_sample T=0")
                print(K)
            k = -np.matmul(np.linalg.inv(Q[6:, 6:]), q[6:])
            V = Q[0:6, 0:6] + np.matmul(Q[0:6, 6:], K) + np.matmul(np.transpose(K), Q[6:, 0:6]) + np.matmul(
                np.matmul(np.transpose(K), Q[6:, 6:]), k)
            v = q[0:6] + np.matmul(Q[0:6, 6:], k) + np.matmul(np.transpose(K), q[6:]) + np.matmul(
                np.matmul(np.transpose(K), Q[6:, 6:]), k)
            self.K_list[i] = K
            self.V_list[i] = V
            self.Q_list[i] = Q
            self.k_list[i] = k
            self.v_list[i] = v
            self.q_list[i] = q

    def _forward(self):
        x = np.array([0.5, 0.4, 0.3, 0.01, 0.01, 0.01])
        self.test_env.reset()
        print("forward")
        for i in range(self.T):
            u = np.matmul(self.K_list[i], x - self.xhat[i]) + self.k_list[i] + self.uhat[i]
            self.xhat[i] = x
            self.uhat[i] = u
            x, _, __, ___ = self.test_env.step(u)
            #print(self.K_list[i])
            #print(self.xhat[i])
            #print(self.uhat[i])

    def run(self, lr_rate=1e-4):
        summary_writer = tf.summary.FileWriter("./log")
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            print("reset")
            for i in range(self.T):
                self.K_list[i] = np.array(
                    [[-0.5, 0, 0, -0.5, 0, 0], [0, -0.5, 0, 0, -0.5, 0], [0, 0, -0.5, 0, 0, -0.5]])
            self.collect_sample()
            for i in range(500):
                loss, summary_str = self.model.update(lr_rate=lr_rate, summary=True)
                print("iteration={0} loss={1}".format(i, loss))
                summary_writer.add_summary(summary_str, i)

            for i in range(2000):
                self._backward()
                self.collect_sample()
                for j in range(5):
                    if True:  # i % 10 == 0 and j % 10 == 4:
                        loss, summary_str = self.model.update(lr_rate=lr_rate, summary=True)
                        print("iteration={0} loss={1}".format(i, loss))
                        summary_writer.add_summary(summary_str, i)
                    else:
                        self.model.update(lr_rate=lr_rate, summary=False)
                if True:  # i % 100 == 0:
                    self.store_record(i)
                self._forward()
                if i % 50 == 0:
                    lr_rate = lr_rate / 2
