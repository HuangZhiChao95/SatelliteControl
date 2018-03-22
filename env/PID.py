from env.SatelliteEnv import SatelliteEnv
import matplotlib.pyplot as plt
import numpy as np

env = SatelliteEnv({"theta":np.array([[-0.204,-0.255,0.298]]),"wb": np.array([[-0.0040,-0.0041,0.0050]]), "tspan":0.5})
state = env.reset()

for i in range(0,400):
    action = -0.5*state[0:3]-5*state[3:6]
    state, reward, done,_ = env.step(action)
    print(np.dot(action,action)/0.09)
    print("step={0} reward={1}".format(i,reward))

states = np.array(env.state_list)
plt.figure(1)
plt.subplot(221)
for i in range(0,3):
    plt.plot(states[:,i])

plt.subplot(222)
for i in range(3,6):
    plt.plot(states[:,i])

plt.subplot(223)
for i in range(6,9):
    plt.plot(states[:,i])

#

plt.show()