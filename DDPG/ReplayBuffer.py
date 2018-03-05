import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size=100000, shape=(7,3,7,1)):
        self._queue=[0 for i in range(0,size)]
        self._head = 0
        self._size = size
        self._full = False
        self._shape = shape

    def push(self, states):
        if type(states)!=list:
            states=[states]
        for state in states:
            self._queue[self._head]=state
            self._head = (self._head+1)%self._size
            if self._head==0:
                self._full=True

    def sample(self, number):
        t1 = np.ndarray((number, self._shape[0]), dtype=np.float32)
        t2 = np.ndarray((number, self._shape[1]), dtype=np.float32)
        t3 = np.ndarray((number, self._shape[2]), dtype=np.float32)
        t4 = np.ndarray((number, self._shape[3]), dtype=np.float32)
        if self._full:
            max = self._size
        else:
            max = self._head
        for i in range(0,number):
            index = int(random.uniform(0, max))
            tmp = self._queue[index]
            t1[i] = tmp[0]
            t2[i] = tmp[1]
            t3[i] = tmp[2]
            t4[i] = tmp[3]
        return t1,t2,t3,t4
