import gym
import numpy as np
from math import cos,sin
from scipy.integrate import odeint
from gym import spaces

class SatelliteEnv(gym.Env):

    def _getParameter(self, default, new, key):
        if new.get(key) is not None:
            return new.get(key)
        else:
            return default.get(key)

    def _eulerToq(self, euler):
        euler = euler/2.0
        t1,t2,t3= euler[1], euler[2], euler[3]
        q0 = cos(t2) * cos(t1) * cos(t3) - sin(t2) * sin(t1) * sin(t3)
        q1 = cos(t2) * sin(t1) * cos(t3) - sin(t2) * cos(t1) * sin(t3)
        q2 = sin(t2) * cos(t1) * cos(t3) + cos(t2) * sin(t1) * sin(t3)
        q3 = sin(t2) * sin(t1) * cos(t3) + cos(t2) * cos(t1) * sin(t3)
        return np.array([q0,q1,q2,q3])
    
    def _odefun(self, y, t, T):
        dq = self._wToqhat(y[0:4],y[4:7], self.w0)
        w = y[4:7]
        omega = y[7:10]
        dw = self.Ib_invert*(T-np.cross(w,np.dot(self.Ib,w)+np.dot(np.dot(self.Cw,omega))))
        domega = -self.Cw_invert*T
        return np.concatenate((dq, dw, domega))

    def _wToqhat(self,q,w,w0):
        wx, wy, wz = w[0], w[1], w[2]
        A=np.array([[0,-wx,-wy-w0,-wz],
                    [wx,0,wz,-wy+w0],
                    [wy+w0,-wz,0,wx],
                    [wz,wy-w0,-wx,0]])/2
        return np.dot(A,q)
    
    def __init__(self, parameter=dict()):
        self.defaultParameter={
            "I":np.array([[12.77, -0.366, 0.158],[-0.366, 133, 0.099], [0.158, 0.099, 133]]),
            "Iw": 4.133e-4,
            "hmax":0.3,
            "omega":np.zeros(shape=(3)),
            "theta":np.array([0.5, 0.3, 0.5])*np.pi/180,
            "wb": np.ones(shape=(3))*0.02*np.pi/180,
            "w0":2*np.pi/86400,
            "C":np.eye(3),
            "tspan":1
        }
        self.parameter = parameter
        self.action_space = spaces.Box(-np.ones((3))*0.1, np.ones((3))*0.1)
        self.observation_space = spaces.Box(-np.ones(8)*50, np.ones(8)*50)
        self.reward_range = spaces.Box(np.array([-10]), np.array([0]))


    def _step(self, action):
        t = np.linspace(0, self.tsapn, 10)
        y_init = np.concatenate((self.q, self.wb, self.omega))
        states = odeint(self._odefun, y_init, t, args=action, printmessg=True)
        for state in states[:-1]:
            self.state_list.append(state)

        self.q = states[-1,0:4]
        self.wb = states[-1, 4:7]
        self.omega = self[-1, 7:]

        tmp = self.q - [1,0,0,0]
        reward =  -10 * np.dot(tmp,tmp) - np.dot(action,action)/0.09
        done = False
        if reward<-1 or self.step>1000:
            done = True
        self.step = self.step+1
        return np.concatenate((self.q, self._wToqhat(self.q, self.wb, self.w0))), reward, done, { }


    # def _seed(self, seed = None):
    #     pass

    def _reset(self):
        I = self._getParameter(self.defaultParameter, self.parameter, "I")
        C = self._getParameter(self.defaultParameter, self.parameter, "C")
        Iw = self._getParameter(self.defaultParameter, self.parameter, "Iw")
        self.hmax = self._getParameter(self.defaultParameter, self.parameter, "hmax")
        self.omega = self._getParameter(self.defaultParameter, self.parameter, "omega")
        theta = self._getParameter(self.defaultParameter, self.parameter, "theta")
        self.wb = self._getParameter(self.defaultParameter, self.parameter, "wb")
        self.w0 = self._getParameter(self.defaultParameter, self.parameter, "w0")
        self.tsapn = self._getParameter(self.defaultParameter, self.parameter, "tspan")

        self.Cw = C*Iw
        self.Cw_invert = np.invert(self.Cw)
        Ieig, _ = np.linalg.eig(I)
        self.Ib_invert = np.invert(np.diag(Ieig))
        self.step = 0
        self.q = self._eulerToq(theta)
        self.state_list = list()

    def _render(self, mode='human', close=False):
        pass
