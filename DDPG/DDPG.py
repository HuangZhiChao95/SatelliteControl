import tensorflow as tf
import numpy as np
from DDPG.ReplayBuffer import ReplayBuffer

class DDPG:

    def __init__(self, policy_unit=[50, 50], value_unit=[300, 400], state_dim=7, action_dim=3, policy_init_std=1e-2,
                 value_init_std=1e-1, gamma=0.99, tau=0.01, replaysize=100000, batch_size=64):
        amp = 1e-3
        self.debug = list()

        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="state")
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="next_state")
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="reward")
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action")
        self.lr_rate = tf.placeholder(dtype=tf.float32, name="lr_rate")
        self.flag_update_value = tf.placeholder(dtype=tf.bool, name="flag_update_value")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")
        self.action_pd = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action_pd")

        self.gamma = gamma
        self.tau = tau

        self.replaybuffer = ReplayBuffer(replaysize, shape=(state_dim, action_dim, state_dim, 1))
        self.coupleweight_op = list()
        self.batch_size = batch_size
        self.action_dim = action_dim

        policy_unit = [state_dim] + policy_unit + [action_dim]
        value_unit = [state_dim + action_dim] + value_unit + [1]

        # self.state_norm, self.next_state_norm = self._input_batch_norm_state(state_dim)
        # self.action_norm = self._input_batch_norm_action(action_dim)
        self.state_norm = self._input_batch_norm(self.state, state_dim)
        self.next_state_norm = self._input_batch_norm(self.next_state, state_dim)
        self.action_norm = self._input_batch_norm(self.action, action_dim)

        source_policy = self.state_norm
        target_policy = self.next_state_norm
        for i in range(0, len(policy_unit) - 1):
            if i == len(value_unit) - 2:
                act = tf.identity
            else:
                act = tf.tanh
            with tf.name_scope("policy_fc{0}".format(i + 1)):
                source_policy, target_policy = self._double_denselayer(source_policy, target_policy, policy_unit[i],
                                                                       policy_unit[i + 1], policy_init_std, "policy", act)

        self.source_action = tf.clip_by_value(source_policy, -1e-1, 1e-1)
        self.target_action = tf.clip_by_value(target_policy, -1e-1, 1e-1)
        self.source_action_norm = self._input_batch_norm(self.source_action, action_dim)
        self.target_action_norm = self._input_batch_norm(self.target_action, action_dim)

        source_value = tf.concat((self.state_norm, tf.cond(self.flag_update_value, lambda: self.action_norm, lambda: self.source_action_norm)), axis=1)
        target_value = tf.concat((self.next_state_norm, self.target_action_norm), axis=1)

        for i in range(0, len(value_unit) - 1):
            if i == len(value_unit) - 2:
                act = tf.identity
                bias = True
            else:
                act = tf.nn.relu
                bias = True
            with tf.name_scope("value_fc{0}".format(i + 1)):
                source_value, target_value = self._double_denselayer(source_value, target_value, value_unit[i],
                                                                     value_unit[i + 1], value_init_std, "value", act, bias)

        self.fitQ = self.reward + self.gamma * target_value
        self.debug = target_value
        self.loss = tf.reduce_mean(tf.abs(self.fitQ - source_value))
        self.Q_source=source_value
        self.Q = tf.reduce_mean(source_value)
        self.train_policy = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(-self.Q, var_list=tf.get_collection("policy"))
        self.train_value = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss, var_list=tf.get_collection("value"))
        
        self.action_pd_loss = tf.reduce_mean(tf.abs(self.source_action-self.action_pd))
        self.train_action_pd = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.action_pd_loss, var_list=tf.get_collection("policy"))


    # def _input_batch_norm_state(self, dim):
    #     batch_mean, batch_var = tf.nn.moments(self.state, [0], name='state_moments')
    #     self.debug.append(batch_mean)
    #     self.debug.append(batch_var)
    #     ema = tf.train.ExponentialMovingAverage(decay=0.9)
    #
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)
    #
    #     mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #
    #     beta = tf.zeros(dim, name="state_beta")
    #     gamma = tf.ones(dim, name="state_gamma")
    #     return tf.nn.batch_normalization(self.state, mean, var, beta, gamma, 1e-4), \
    #            tf.nn.batch_normalization(self.next_state, mean, var, beta, gamma, 1e-4)
    #
    # def _input_batch_norm_action(self, dim):
    #     batch_mean, batch_var = tf.nn.moments(self.action, [0], name='action_moments')
    #     ema = tf.train.ExponentialMovingAverage(decay=0.9)
    #     self.debug.append(batch_mean)
    #     self.debug.append(batch_var)
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([batch_mean, batch_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(batch_mean), tf.identity(batch_var)
    #
    #     mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    #
    #     beta = tf.zeros(dim, name="action_beta")
    #     gamma = tf.ones(dim, name="action_gamma")
    #     return tf.nn.batch_normalization(self.action, mean, var, beta, gamma, 1e-4)

    def _input_batch_norm(self, input, dim):
        #return tf.identity(input)
        batch_mean, batch_var = tf.nn.moments(input, [0])
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

        beta = tf.zeros(dim)
        gamma = tf.ones(dim)
        #self.debug.append(ema.average(batch_var))
        tf.summary.histogram(values=ema.average(batch_mean), name="batch_mean")
        tf.summary.histogram(values=ema.average(batch_var), name="batch_var")
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-4)

    def _double_denselayer(self, source, target, n_in, n_out, init_std, collection, activation=tf.nn.tanh, bias=False):
        w_source = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=init_std), name="source_weight",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        b_source = tf.Variable(tf.zeros([n_out]), name="source_bias",
                               collections=[collection, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        if bias:
            source = activation(tf.nn.bias_add(tf.matmul(source, w_source), b_source), name="source")
        else:
            source = activation(tf.matmul(source, w_source), name="source")
        tf.summary.histogram(values=w_source, name="source_weight")
        tf.summary.histogram(values=b_source, name="source_bias")
        #tf.summary.histogram(values=source, name="source")

        w_target = tf.Variable(w_source, name="target_weight")
        b_target = tf.Variable(b_source, name="target_bias")
        if bias:
            target = activation(tf.nn.bias_add(tf.matmul(target, w_target), b_target), name="target")
        else:
            target = activation(tf.matmul(target, w_target), name="target")
        tf.summary.histogram(values=w_target, name="target_weight")
        tf.summary.histogram(values=b_target, name="target_bias")
        #tf.summary.histogram(values=target, name="target")
        self.coupleweight_op.append(tf.assign(w_target, self.tau * w_source + (1 - self.tau) * w_target))
        self.coupleweight_op.append(tf.assign(b_target, self.tau * b_source + (1 - self.tau) * b_target))

        return source, target

    def predict(self, state, sess):
        action = sess.run(self.target_action, feed_dict={self.next_state: state, self.phase_train: False})
        return action

    def predict_noise(self, state, sess, std=1e-3):
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

    def update(self, lr_rate, sess, imitate_pd=False):
        state, action, next_state, reward = self.replaybuffer.sample(self.batch_size)
        value_feed_dict = {
            self.state: state,
            self.next_state: next_state,
            self.lr_rate: lr_rate*100,
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
        _, loss, fitQ, Q_all = sess.run([self.train_value, self.loss, self.fitQ, self.Q_source], feed_dict=value_feed_dict)
        #print(np.mean(debug))
        if imitate_pd:
            self.tau=0.005
            feed_dict = {
                self.state: state,
                self.lr_rate: lr_rate*10,
                self.action_pd: -0.5*state[:,1:4]-0.5*state[:,4:7],
                self.flag_update_value: False,
                self.phase_train: True
            }
            _, pd_loss = sess.run([self.train_action_pd, self.action_pd_loss], feed_dict = feed_dict)
        else:
            _, Q = sess.run([self.train_policy, self.Q], feed_dict=policy_feed_dict)
        #Q = sess.run(self.Q, feed_dict=policy_feed_dict)

        # debug = sess.run([self.target_action_norm, self.target_action]+self.debug, feed_dict={self.state: state, self.action:action, self.next_state: next_state, self.phase_train:False})
        # print(debug)

        for op in self.coupleweight_op:
            sess.run(op)
        if imitate_pd:
            return loss,pd_loss,fitQ,reward,Q_all
        else:
            return loss, Q, fitQ, reward, Q_all

