import gym
import numpy as np
from math import cos,sin,exp
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
        t1,t2,t3= euler[0], euler[1], euler[2]
        q0 = cos(t2) * cos(t1) * cos(t3) - sin(t2) * sin(t1) * sin(t3)
        q1 = cos(t2) * sin(t1) * cos(t3) - sin(t2) * cos(t1) * sin(t3)
        q2 = sin(t2) * cos(t1) * cos(t3) + cos(t2) * sin(t1) * sin(t3)
        q3 = sin(t2) * sin(t1) * cos(t3) + cos(t2) * cos(t1) * sin(t3)
        return np.array([q0,q1,q2,q3])
    
    def _odefun(self, y, t, T):
        dq = self._wToqhat(y[0:4],y[4:7], self.w0)
        w = y[4:7]
        omega = y[7:10]
        dw = np.dot(self.Ib_inverse,T-np.cross(w,np.dot(self.Ib,w)+np.dot(self.Cw,omega)))
        domega = -np.dot(self.Cw_inverse,T)
        return np.concatenate((dq, dw, domega))

    def _wToqhat(self,q,w,w0):
        Abo = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                        [2*(q[1]*q[2]-q[0]*q[3]) , q[0]**2-q[1]**2+q[2]**2-q[3]**2 , 2*(q[2]*q[3]+q[0]*q[1])],
                        [2*(q[0]*q[3]+q[0]*q[2]) , 2*(q[2]*q[3]-q[1]*q[0] ), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
        woi = [0,w0,0]
        woib = np.dot(Abo, woi)
        q0 = q[0]
        q1 = q[1:4]
        wbo = w-woib
        dq0 = -0.5*np.dot(q1,wbo)
        dq1 = np.cross(q1,wbo)+q0*wbo/2;
        return np.array([dq0,dq1[0],dq1[1],dq1[2]])
    
    def __init__(self, parameter=dict()):
        self.defaultParameter={
            "I":np.array([[12.77, -0.366, 0.158],[-0.366, 133, 0.099], [0.158, 0.099, 133]]),
            "Iw": 4.133e-4,
            "hmax":0.3,
            "omega":np.zeros(shape=(3)),
            "theta":np.array([0.5, 0.3, 0.5])*np.pi/180,
            "wb": np.ones(shape=(3))*0.02*np.pi/180,
            "w0":0.001097231046810,
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
        states = odeint(self._odefun, y_init, t, args=(action,), printmessg=True)
        for state in states[:-1]:
            self.state_list.append(state)

        self.q = states[-1,0:4]
        self.wb = states[-1, 4:7]
        self.omega = states[-1, 7:]

        tmp = self.q - [1,0,0,0]
        reward =  (-10 * np.dot(tmp,tmp) - np.dot(action,action)*20)*exp(self.step_count*self.tsapn/500)
        done = False
        self.step_count = self.step_count+1
        if self.step_count*self.tsapn>1000 or reward<-40:
            done = True
            
        return np.concatenate((self.q, self.wb)), reward, done, { }


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
        self.Cw_inverse = np.linalg.inv(self.Cw)
        Ieig, _ = np.linalg.eig(I)
        self.Ib = np.diag(Ieig)
        self.Ib_inverse = np.diag(1/Ieig)
        self.step_count = 0
        self.q = self._eulerToq(theta)
        self.state_list = list()
        return np.concatenate((self.q, self.wb))

    def _render(self, mode='human', close=False):
        pass
