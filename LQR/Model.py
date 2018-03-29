import tensorflow as tf
import numpy as np
from sklearn import linear_model

class Model:
    def __init__(self, state_dim=6, action_dim=3, tspan=0.5, w0=0.001097231046810):
        self.state_dim = state_dim
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, state_dim + action_dim], name="input")
        self.w_linear = tf.placeholder(dtype=tf.float32, shape=[6, action_dim], name="w_linear")
        self.w_square = tf.placeholder(dtype=tf.float32, shape=[6, action_dim], name="w_square")
        self.w_bias = tf.placeholder(dtype=tf.float32, shape=[action_dim], name="w_bias")
        self.tspan = tspan
        self.w0 = w0
        self.action_dim = action_dim
        self.linear_fit = linear_model.LinearRegression()

        q = tf.slice(self.input, [0, 0], [-1, 3])
        w = tf.slice(self.input, [0, 3], [-1, 3])
        a = tf.slice(self.input, [0, 6], [-1, 3])
        w_a = tf.slice(self.input, [0, 3], [-1, 6])

        q_next = self._q_model(q, w, w0) + q
        w_next = w + tf.nn.bias_add(tf.matmul(w_a, self.w_linear) + tf.matmul(w_a, self.w_square), self.w_bias)
        next_state_linear = tf.concat([q_next, w_next], axis=1)
        self.next_state_model = next_state_linear
        self.jocobi = []
        for i in range(state_dim):
            tmp = tf.gradients(tf.slice(self.next_state_model, [0, i], [-1, 1]), self.input)
            tf.summary.histogram(values = tmp, name="jocobi_{0}".format(i))
            self.jocobi.append(tmp)

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

    def predict(self, state, action, sess=None):
        input = np.concatenate((state[np.newaxis, :], action[np.newaxis, :]), axis=1)
        feed_dict = {
            self.input: input,
            self.w_linear: self.linear_fit.coef_[:,:6],
            self.w_square: self.linear_fit.coef_[:,6:],
            self.w_bias: self.linear_fit.intercept_
        }
        sess = sess or tf.get_default_session()
        next_state = sess.run(self.next_state_model, feed_dict=feed_dict)
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

    def update(self):
        w_a = np.concatenate((self.states[:,3:],self.actions),axis=1)
        self.linear_fit.fit(np.concatenate((w_a, w_a*w_a), axis=1), self.next_states[:, 3:6] - self.states[:, 3:6])
