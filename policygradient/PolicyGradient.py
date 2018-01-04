import tensorflow as tf
import tensorlayer as tl
import numpy as np

class PolicyGradient:

    def __init__(self, hidden_unit=[200,200], input_dim=7, output_dim=3, init_std = 1e-3, gamma=0.9999, batch_size=64):
        amp = 1
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])/amp;
        self.lr_rate = tf.placeholder(dtype=tf.float32)
        self.std = tf.placeholder(dtype=tf.float32)
        self.gamma=gamma

        network = tl.layers.InputLayer(self.input)
        for i, unit_num in enumerate(hidden_unit):
            network = tl.layers.DenseLayer(network, n_units=unit_num, act=tf.nn.tanh, W_init=tf.truncated_normal_initializer(stddev=init_std), name="dense{0}".format(i))

        mean = tl.layers.DenseLayer(network, n_units=output_dim, W_init=tf.truncated_normal_initializer(stddev=init_std), name="mean").outputs
        #std = tl.layers.DenseLayer(network, n_units=output_dim, act=tf.nn.softplus, W_init=tf.truncated_normal_initializer(stddev=init_std), name="std").outputs + 1e-5
        #k = tf.Variable(-0.05)
        #mean = k*(tf.slice(self.input,[0,1],[-1,3])+tf.slice(self.input,[0,4],[-1,3]))

        train_params = network.all_params
        normal_dist = tf.contrib.distributions.Normal(mean, self.std)
        self.action_predict = tf.squeeze(normal_dist.sample([1])) * amp
        log_prob = tf.reduce_prod(tf.squeeze(normal_dist.log_prob(self.action)), axis=1)
        self.loss = tf.reduce_sum(tf.multiply(-log_prob, self.target), axis=0) / batch_size
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss, var_list=train_params)
        tf.summary.scalar('loss', self.loss)

        self.action_list = list()
        self.state_list = list()
        self.merge_op = tf.summary.merge_all()
        #self.oplist = [k]

    def _clearMem(self):
        self.action_list.clear()
        self.state_list.clear()

    def clearMem(self):
        self._clearMem()

    def _memorize(self, action, state):
        self.action_list.append(action.copy())
        self.state_list.append(state.copy())

    def predict(self, state, sess, std=1e-4):
        action = sess.run(self.action_predict, {self.input: state, self.std: std})
        self._memorize(action, state)
        return action

    def debug(self, state, sess):
        return sess.run(self.oplist, {self.input: state})

    def update(self, reward, lr_rate, sess):
        reward_array = np.array(reward)
        target = np.zeros(reward_array.shape, dtype=np.float32)
        temp = np.zeros(target.shape[1], dtype=np.float32)
        i = len(reward) - 1
        while i>=0:
            temp  = self.gamma*temp + reward[i]
            target[i] = temp - np.mean(temp)
            i = i-1

        target = target.reshape(-1)
        action_array = np.array(self.action_list)
        action_array = action_array.reshape((-1,action_array.shape[-1]))
        state_array = np.array(self.state_list)
        state_array = state_array.reshape((-1,state_array.shape[-1]))
        feed_dict = {
            self.input: state_array,
            self.action: action_array,
            self.target: target,
            self.lr_rate:lr_rate,
            self.std: 1e-4
        }

        _, loss, summary_str = sess.run([self.train_op, self.loss, self.merge_op], feed_dict=feed_dict)
        #print(loss)
        self._clearMem()
        return temp, loss, summary_str








