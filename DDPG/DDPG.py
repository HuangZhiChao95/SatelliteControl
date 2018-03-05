import tensorflow as tf
import numpy as np
from DDPG.ReplayBuffer import ReplayBuffer

class DDPG:

    def __init__(self, policy_unit=[200, 200], value_unit=[200, 200], state_dim=7, action_dim=3, policy_init_std=1e-3,
                 value_init_std=1e-3, gamma=0.99, tau=0.99, replaysize=100000, batch_size=64):
        amp = 1
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="state")
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="next_state")
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="reward")
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action")
        self.lr_rate = tf.placeholder(dtype=tf.float32, name="lr_rate")
        self.flag_update_value = tf.placeholder(dtype=tf.bool, name="flag_update_value")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")

        self.gamma = gamma
        self.tau = tau

        self.replaybuffer = ReplayBuffer(replaysize, shape=(state_dim, action_dim, state_dim, 1))
        self.coupleweight_op = list()
        self.batch_size = batch_size
        self.action_dim = action_dim

        policy_unit = [state_dim] + policy_unit + [action_dim]
        value_unit = [state_dim + action_dim] + value_unit + [1]

        self.state_norm, self.next_state_norm = self._input_batch_norm_state(state_dim)
        self.action_norm = self._input_batch_norm_action(action_dim)

        source_policy = self.state_norm
        target_policy = self.next_state_norm
        for i in range(0, len(policy_unit) - 1):
            with tf.name_scope("policy_fc{0}".format(i + 1)):
                source_policy, target_policy = self._double_denselayer(source_policy, target_policy, policy_unit[i],
                                                                       policy_unit[i + 1], policy_init_std, "policy")

        self.source_action = source_policy
        self.target_action = target_policy

        source_value = tf.concat((self.state_norm, tf.cond(self.flag_update_value, lambda: self.action_norm, lambda: self.source_action)), axis=1)
        target_value = tf.concat((self.next_state_norm, self.target_action), axis=1)
        for i in range(0, len(value_unit) - 1):
            with tf.name_scope("value_fc{0}".format(i + 1)):
                source_value, target_value = self._double_denselayer(source_value, target_value, value_unit[i],
                                                                     value_unit[i + 1], value_init_std, "value")

        self.fitQ = self.reward + self.gamma * target_value
        self.loss = tf.reduce_mean(tf.square(self.fitQ - source_value))
        self.Q = tf.reduce_mean(source_value)
        print(tf.get_collection("policy"))
        print(tf.get_collection("value"))
        self.train_policy = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(-self.Q, var_list=tf.get_collection("policy"))
        self.train_value = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss, var_list=tf.get_collection("value"))

    def _input_batch_norm_state(self, dim):
        batch_mean, batch_var = tf.nn.moments(self.state, [0], name='state_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        beta = tf.zeros(dim, name="state_beta")
        gamma = tf.ones(dim, name="state_gamma")
        return tf.nn.batch_normalization(self.state, mean, var, beta, gamma, 1e-4), \
               tf.nn.batch_normalization(self.next_state, mean, var, beta, gamma, 1e-4)

    def _input_batch_norm_action(self, dim):
        batch_mean, batch_var = tf.nn.moments(self.action, [0], name='action_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        beta = tf.zeros(dim, name="action_beta")
        gamma = tf.ones(dim, name="action_gamma")
        return tf.nn.batch_normalization(self.action, mean, var, beta, gamma, 1e-4)

    def _double_denselayer(self, source, target, n_in, n_out, init_std, collection, activation=tf.nn.relu):
        w_source = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=init_std), name="source_weight",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        b_source = tf.Variable(tf.zeros([n_out]), name="source_bias",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        source = activation(tf.nn.bias_add(tf.matmul(source, w_source), b_source), name="source")
        tf.summary.histogram(values=w_source, name="source_weight")
        tf.summary.histogram(values=b_source, name="source_bias")
        tf.summary.histogram(values=source, name="source")

        w_target = tf.Variable(w_source, name="target_weight", trainable=False)
        b_target = tf.Variable(b_source, name="target_bias", trainable=False)
        target = activation(tf.nn.bias_add(tf.matmul(target, w_target), b_target), name="target")
        tf.summary.histogram(values=w_target, name="target_weight")
        tf.summary.histogram(values=b_target, name="target_bias")
        tf.summary.histogram(values=target, name="target")

        self.coupleweight_op.append(tf.assign(w_target, self.tau * w_source + (1 - self.tau) * w_target))
        self.coupleweight_op.append(tf.assign(b_target, self.tau * b_source + (1 - self.tau) * b_target))

        return source, target

    def predict(self, state, sess):
        action = sess.run(self.target_action, feed_dict={self.next_state: state, self.phase_train: False})
        return action

    def predict_noise(self, state, sess, std=1e-4):
        feed_dict = {
            self.state: state,
            self.phase_train: False
        }
        actions = sess.run(self.source_action, feed_dict=feed_dict)
        actions += np.random.randn(actions.shape[0], self.action_dim) * std
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
            self.flag_update_value: True,
            self.phase_train: True
        }
        policy_feed_dict = {
            self.state: state,
            self.lr_rate: lr_rate,
            self.action: action,
            self.flag_update_value: False,
            self.phase_train: True
        }
        _, loss = sess.run([self.train_value, self.loss], feed_dict=value_feed_dict)
        _, Q = sess.run([self.train_policy, self.Q], feed_dict=policy_feed_dict)

        for op in self.coupleweight_op:
            sess.run(op)

        return loss, Q
