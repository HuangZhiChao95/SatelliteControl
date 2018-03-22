import gym
import numpy as np
from math import cos, sin, exp, tan
from scipy.integrate import odeint
from gym import spaces


class SatelliteEnv(gym.Env):
    def _getParameter(self, default, new, key):
        if new.get(key) is not None:
            return new.get(key)
        else:
            return default.get(key)

    def _odefun(self, y, t, T):
        dtheta = self._wTothat(y[0:3], y[3:6], self.w0)
        w = y[3:6]
        omega = y[6:9]
        dw = np.dot(self.Ib_inverse, T - np.cross(w, np.dot(self.Ib, w) + np.dot(self.Cw, omega)))
        domega = -np.dot(self.Cw_inverse, T)
        # print("{0} {1}".format(t,T))
        return np.concatenate((dtheta, dw, domega))

    def _wTothat(self, theta, w, w0):
        t1 = theta[0]
        t2 = theta[1]
        t3 = theta[2]
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        dt1 = w1 * cos(t2) + w3 * sin(t2) + w0 * sin(t3)
        dt2 = w2 - tan(t1) * (w3 * cos(t2) - w1 * sin(t2)) + w0 * cos(t3) / cos(t1)
        dt3 = (w3 * cos(t2) - w1 * sin(t2) - w0 * sin(t1) * cos(t3)) / cos(t1)
        #print("{0}_{1}".format(dt2,w2))
        return np.array([dt1, dt2, dt3])

    def __init__(self, parameter=dict(), debug=False):
        self.defaultParameter = {
            "I": np.array([[12.77, -0.366, 0.158], [-0.366, 133, 0.099], [0.158, 0.099, 133]]),
            "Iw": 4.133e-4,
            "hmax": 0.3,
            "omega": np.zeros(shape=(3)),
            "theta": np.ones((5000, 3), dtype=np.float32) * 0.5 * np.pi / 180,
            "wb": np.ones((5000, 3), dtype=np.float32) * 0.02 * np.pi / 180,
            "w0": 0.001097231046810,
            "C": np.eye(3),
            "tspan": 1
        }
        self.parameter = parameter
        self.action_space = spaces.Box(-np.ones((3)) * 0.1, np.ones((3)) * 0.1)
        self.observation_space = spaces.Box(-np.ones(8) * 50, np.ones(8) * 50)
        self.reward_range = spaces.Box(np.array([-10]), np.array([0]))
        self.iteration = 0
        self.wb_list = self._getParameter(self.defaultParameter, self.parameter, "wb")
        self.theta_list = self._getParameter(self.defaultParameter, self.parameter, "theta")
        self.debug = debug

    def _step(self, action):
        t = np.linspace(0, self.tsapn, 10)
        y_init = np.concatenate((self.theta, self.wb, self.omega))
        states = odeint(self._odefun, y_init, t, args=(action,), printmessg=True)
        for state in states[:-1]:
            self.state_list.append(state)

        self.theta = states[-1, 0:3]
        self.wb = states[-1, 3:6]
        self.omega = states[-1, 6:9]
        #self.stheta = states[-1, 9:12]

        tmp = self.theta
        reward = (- np.sum(np.abs(tmp)) - np.sum(np.abs(action)))
        done = False
        self.step_count = self.step_count + 1
        if (self.step_count * self.tsapn > 1000 or reward < -0.5) and self.step_count * self.tsapn > 500:
            done = True

        return np.concatenate((self.theta, self.wb)), reward, done, {}

    def _reset(self):
        I = self._getParameter(self.defaultParameter, self.parameter, "I")
        C = self._getParameter(self.defaultParameter, self.parameter, "C")
        Iw = self._getParameter(self.defaultParameter, self.parameter, "Iw")
        self.hmax = self._getParameter(self.defaultParameter, self.parameter, "hmax")
        self.omega = self._getParameter(self.defaultParameter, self.parameter, "omega")
        theta = self._getParameter(self.defaultParameter, self.parameter, "theta")
        self.w0 = self._getParameter(self.defaultParameter, self.parameter, "w0")
        self.tsapn = self._getParameter(self.defaultParameter, self.parameter, "tspan")

        theta = self.theta_list[self.iteration, :]
        self.wb = self.wb_list[self.iteration, :]
        self.iteration = (self.iteration + 1) % len(self.theta_list)

        self.Cw = C * Iw
        self.Cw_inverse = np.linalg.inv(self.Cw)
        Ieig, _ = np.linalg.eig(I)
        self.Ib = np.diag(Ieig)
        self.Ib_inverse = np.diag(1 / Ieig)
        self.step_count = 0
        self.theta = theta
        self.state_list = list()
        self.stheta = np.zeros(3, dtype=np.float32)
        return np.concatenate((self.theta, self.wb))

    def _render(self, mode='human', close=False):
        pass
