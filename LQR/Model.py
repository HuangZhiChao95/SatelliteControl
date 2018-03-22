import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, unit=[128], state_dim=6, action_dim=3, init_std=1e-2, act=tf.nn.relu,
                 tspan=0.5, w0=0.001097231046810):
        self.state_dim = state_dim
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim + action_dim], name="input")
        self.output = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="action")
        self.lr_rate = tf.placeholder(dtype=tf.float32, name="lr_rate")
        self.phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")
        self.w_weight = tf.placeholder(dtype=tf.float32, name="w_weight")
        self.tspan = tspan
        self.w0 = w0
        self.action_dim = action_dim

        unit = [state_dim + action_dim] + unit + [state_dim]
        with tf.name_scope("input"):
            network = self._batch_norm(self.input, state_dim + action_dim)

        for i in range(0, len(unit) - 1):
            if i == len(unit) - 2:
                act = tf.identity
                batch_norm = False
            else:
                batch_norm = True
            with tf.name_scope("dense{0}".format(i + 1)):
                network = self._denselayer(network, unit[i], unit[i + 1], init_std, act, batch_norm)
        # network = network/10
        q = tf.slice(self.input, [0, 0], [-1, 3])
        w = tf.slice(self.input, [0, 3], [-1, 3])
        a = tf.slice(self.input, [0, 6], [-1, 3])
        q_next = self._q_model(q, w, w0) + q  # self._linear_op(a, 0.05, 2, "kq2") + q
        w_next = self._linear_op(a, 0.1, 1, "kw1") + w  # self._linear_op(a, 0.05, 2, "kw2") + w
        next_state_linear = tf.concat([q_next, w_next], axis=1)
        self.next_state_model = network + next_state_linear
        # print(self.output.shape)
        # print(self.next_state_model.shape)
        diff = tf.abs(self.next_state_model - self.output)
        diff_q = tf.reduce_mean(tf.slice(diff, [0, 0], [-1, 3]))
        diff_w = tf.reduce_mean(tf.slice(diff, [0, 3], [-1, 3])) * self.w_weight
        self.loss = diff_q + diff_w
        tf.summary.scalar(tensor=self.loss, name="loss")
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss)
        self.jocobi = []
        for i in range(state_dim):
            tmp = tf.gradients(tf.slice(self.next_state_model, [0, i], [-1, 1]), self.input)
            tf.summary.histogram(values = tmp, name="jocobi_{0}".format(i))
            self.jocobi.append(tmp)

    def _linear_op(self, input, init, p, name):
        k = tf.Variable(init * self.tspan, name=name)
        tf.summary.scalar(tensor=k, name=name)
        tmp = input
        for i in range(p-1):
            tmp = tf.multiply(tmp, input)
        return k * tmp

    def _q_model(self, theta, w, w0):
        t1 = tf.slice(theta, [0, 0], [-1, 1])
        t2 = tf.slice(theta, [0, 1], [-1, 1])
        t3 = tf.slice(theta, [0, 2], [-1, 1])
        w1 = tf.slice(w, [0, 0], [-1, 1])
        w2 = tf.slice(w, [0, 1], [-1, 1])
        w3 = tf.slice(w, [0, 2], [-1, 1])

        theta1 = w1 * tf.cos(t2) + w2 * tf.sin(t2) + w0 * tf.sin(t3) * self.tspan
        theta2 = w2 - tf.tan(t1) * (w3 * tf.cos(t2) - w1 * tf.sin(t2)) + w0 * tf.cos(t3) / tf.cos(t1) * self.tspan
        theta3 = (w3 * tf.cos(t2) - w1 * tf.sin(t2) - w0 * tf.sin(t1) * tf.cos(t3)) / tf.cos(t1) * self.tspan

        return tf.concat([theta1,theta2,theta3], axis=1)

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

    def getJocobi(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        input = np.concatenate((state[np.newaxis, :], action[np.newaxis, :]), axis=1)
        gradient_list = sess.run(self.jocobi, feed_dict={self.input: input, self.phase_train: False})
        joboci = np.array(gradient_list)
        return joboci.squeeze()

    def store(self, states, actions, next_states):
        self.states = states.copy()
        self.actions = actions.copy()
        self.next_states = next_states.copy()

    def update(self, lr_rate, sess=None, summary=False):
        index = (np.random.rand(128) * len(self.states)).astype(np.int32)
        state = self.states[index]
        action = self.actions[index]
        next_state = self.next_states[index]
        sess = sess or tf.get_default_session()
        merge_op = tf.summary.merge_all()
        feed_dict = {
            self.input: np.concatenate((state, action), axis=1),
            self.output: next_state,
            self.lr_rate: lr_rate,
            self.phase_train: True,
            self.w_weight: 10
        }
        # print((next_state[:,0:3]-state[:,0:3])/state[:,3:6])
        # print((next_state[:,3:6]-state[:,3:6])/action)
        if summary:
            _, loss, summary_str, debug = sess.run([self.train_op, self.loss, merge_op, self.next_state_model],
                                                   feed_dict=feed_dict)
            #loss, summary_str, debug = sess.run([self.loss, merge_op, self.next_state_model],
            #                                       feed_dict=feed_dict)
            #print(debug)
            return loss, summary_str
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
