from env.SatelliteEnv import SatelliteEnv
import matplotlib.pyplot as plt
import numpy as np

env = SatelliteEnv({"theta":np.array([[1,1,1]]),"tspan":1})
state = env.reset()

for i in range(0,1000):
    action = -0.5*state[1:4]-5*state[4:7]-0.002*state[8:]
    state, reward, done,_ = env.step(action)
    print(np.dot(action,action)/0.09)
    print("step={0} reward={1}".format(i,reward))

states = np.array(env.state_list)
plt.figure(1)
plt.subplot(231)
plt.plot(states[:,0])
plt.subplot(234)
for i in range(1,4):
    plt.plot(states[:,i])

plt.subplot(232)
for i in range(4,7):
    plt.plot(states[:,i])

plt.subplot(235)
for i in range(7,10):
    plt.plot(states[:,i])

plt.subplot(233)
plt.plot(states[:,10])

plt.subplot(236)
for i in range(11,14):
    plt.plot(states[:,i])

plt.show()
