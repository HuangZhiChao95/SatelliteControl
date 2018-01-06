import numpy as np
from multiprocessing import Process, Semaphore, Queue
from env.SatelliteEnv import SatelliteEnv

def envblock(n, parameter, lockself, lockmain, q, state_dim = 7):
    envs = list()
    states = np.zeros([n, state_dim], dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.ndarray(n, dtype=np.bool)

    for i in range(n):
        envs.append(SatelliteEnv(parameter))

    while True:
        lockself.acquire()
        temp = q.get()
        if type(temp) is str:
            for i in range(n):
                states[i,:] = envs[i].reset()
            q.put(states)

        if type(temp) is np.ndarray:
            for i in range(n):
                states[i, :], rewards[i] , dones[i], __ = envs[i].step(action=temp[i, :])
            q.put((states, rewards, dones))
        lockmain.release()





