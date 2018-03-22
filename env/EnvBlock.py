import numpy as np
from multiprocessing import Process, Semaphore, Queue
from env.SatelliteEnv import SatelliteEnv

def envblock(n, parameter, lockself, lockmain, q, state_dim = 6):
    envs = list()
    states = np.zeros([n, state_dim], dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.ndarray(n, dtype=np.bool)

    for i in range(n):
        penv = {
            "tspan": parameter["tspan"],
            "theta": np.matmul(2 * np.random.rand(5000, 3) - 1, parameter["theta"]),
            "wb": np.matmul(2 * np.random.rand(5000, 3) - 1, parameter["wb"])
        }
        envs.append(SatelliteEnv(penv))

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





