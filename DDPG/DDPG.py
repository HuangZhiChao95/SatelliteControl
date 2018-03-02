import tensorflow as tf
import numpy as np
from DDPG.ReplayBuffer import ReplayBuffer

class DDPG:
    def __init__(self, policy_unit=[200, 200], value_unit=[200, 200], state_dim=7, action_dim=3, policy_init_std=1e-3,
                 value_init_std=1e-3, gamma=0.99, tau=0.99, replaysize=100000, batch_size=64):
        amp = 1
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        self.lr_rate = tf.placeholder(dtype=tf.float32)
        self.flag_update_value = tf.placeholder(dtype=tf.bool)
        self.gamma = gamma
        self.tau = tau
        self.replaybuffer = ReplayBuffer(replaysize)
        self.coupleweight_op = list()
        self.batch_size = batch_size
        self.action_dim = action_dim
        policy_unit = [state_dim] + policy_unit + [action_dim]
        value_unit = [state_dim + action_dim] + value_unit + [1]

        source_policy = self.state
        target_policy = self.next_state
        for i in range(0, len(policy_unit) - 1):
            with tf.name_scope("policy_fc{0}".format(i + 1)):
                source_policy, target_policy = self._double_denselayer(source_policy, target_policy, policy_unit[i],
                                                                       policy_unit[i + 1], policy_init_std, "policy")

        self.source_action = tf.cond(self.flag_update_value, lambda: self.action, lambda: source_policy)
        self.target_action = target_policy

        source_value = tf.concat((self.state, self.source_action), axis=1)
        target_value = tf.concat((self.state, self.target_action), axis=1)
        for i in range(0, len(value_unit) - 1):
            with tf.name_scope("value_fc{0}".format(i + 1)):
                source_value, target_value = self._double_denselayer(source_value, target_value, value_unit[i],
                                                                     value_unit[i + 1], value_init_std, "value")

        self.fitQ = self.reward + self.gamma * target_value
        self.loss = tf.reduce_mean(tf.square(self.fitQ - source_value))
        self.Q = tf.reduce_mean(source_value)
        self.train_policy = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(-self.Q, var_list="policy")
        self.train_value = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss, var_list="value")

    def _double_denselayer(self, source, target, n_in, n_out, init_std, collection, activation=tf.nn.relu):
        w_source = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=init_std), name="source_weight",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES])
        b_source = tf.Variable(tf.zeros([n_out]), name="source_bias",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES])
        source = activation(tf.nn.bias_add(tf.matmul(source, w_source), b_source), name="source")
        tf.summary.histogram(w_source, "source_weight")
        tf.summary.histogram(b_source, "source_bias")
        tf.summary.histogram(source, "source")

        w_target = tf.Variable(w_source, name="target_weight", trainable=False)
        b_target = tf.Variable(b_source, name="target_bias", trainable=False)
        target = activation(tf.nn.bias_add(tf.matmul(target, w_target), b_target), name="target")
        tf.summary.histogram(w_target, "target_weight")
        tf.summary.histogram(b_target, "target_bias")
        tf.summary.histogram(target, "target")

        self.coupleweight_op.append(tf.assign(w_target, self.tau * w_source + (1 - self.tau) * w_target))
        self.coupleweight_op.append(tf.assign(b_target, self.tau * b_source + (1 - self.tau) * b_target))

        return source, target

    def predict(self, state, sess):
        action = sess.run(self.target_action, feed_dict={self.state: state})
        return action

    def predict_noise(self, state, sess, std=1e-4):
        feed_dict = {
            self.state: state
        }
        actions = sess.run(self.source_action, feed_dict=feed_dict)
        actions += np.random.randn((self.batch_size, self.action_dim)) * std
        return actions

    def store_sample(self, state, next_state, action, reward):
        for i in range(0, len(reward)):
            self.replaybuffer.push((state[i], action[i], next_state[i], reward[i]))

    def update(self, lr_rate, sess):
        state, action, next_state, reward = self.replaybuffer.sample(self.batch_size)
        value_feed_dict = {
            self.state: state,
            self.next_state: next_state,
            self.lr_rate: lr_rate,
            self.reward: reward,
            self.action: action,
            self.flag_update_value: True
        }
        policy_feed_dict = {
            self.state: state,
            self.lr_rate: lr_rate,
            self.flag_update_value: False
        }
        _, loss = sess.run([self.train_value, self.loss], feed_dict=value_feed_dict)
        _, Q = sess.run([self.train_policy, self.Q], feed_dict=policy_feed_dict)

        for op in self.coupleweight_op:
            sess.run(op)

        return loss, Q
