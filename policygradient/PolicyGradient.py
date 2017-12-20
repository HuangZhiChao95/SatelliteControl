import tensorflow as tf
import tensorlayer as tl
import numpy as np

class PolicyGradient:

    def __init__(self, hidden_unit=[200,200], input_dim=7, output_dim=3, init_std = 1e-3, gamma=0.99):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])
        self.lr_rate = tf.placeholder(dtype=tf.float32)
        self.gamma=gamma

        network = tl.layers.InputLayer(self.input)
        for i, unit_num in enumerate(hidden_unit):
            network = tl.layers.DenseLayer(network, n_units=unit_num, act=tf.nn.tanh, W_init=tf.truncated_normal_initializer(stddev=init_std), name="dense{0}".format(i))

        mean = tl.layers.DenseLayer(network, n_units=output_dim, W_init=tf.truncated_normal_initializer(stddev=init_std), name="mean").outputs
        std = tl.layers.DenseLayer(network, n_units=output_dim, act=tf.nn.softplus, W_init=tf.truncated_normal_initializer(stddev=init_std), name="std").outputs + 1e-5
        train_params = network.all_params
        normal_dist = tf.contrib.distributions.Normal(mean, std)
        self.action_predict = tf.squeeze(normal_dist.sample([1]))
        log_prob = tf.reduce_prod(tf.squeeze(normal_dist.log_prob(self.action)), axis=1)
        self.loss = tf.reduce_mean(tf.multiply(-log_prob, self.target), axis=0)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss, var_list=train_params)

        self.action_list = list()
        self.state_list = list()

    def _clearMem(self):
        self.action_list.clear()
        self.state_list.clear()

    def _memorize(self, action, state):
        self.action_list.append(action)
        self.state_list.append(state)

    def predict(self, state, sess):
        action = sess.run(self.action_predict, {self.input: state})
        self._memorize(action, state)
        return action

    def update(self, reward, lr_rate, sess):
        target = np.zeros(len(reward[0]), dtype=np.float32)
        i = len(reward) - 1
        while i>0:
            target = self.gamma*target + reward[i]
            t = target - np.mean(target)
            feed_dict = {
                self.input: self.state_list[i],
                self.action: self.action_list[i],
                self.target: t,
                self.lr_rate:lr_rate
            }
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            i = i-1

        self._clearMem()
        return target








