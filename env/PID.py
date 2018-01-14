from env.SatelliteEnv import SatelliteEnv
import matplotlib.pyplot as plt
import numpy as np

env = SatelliteEnv({"theta":np.array([[0.6,0.6,0.6]]),"tspan":1})
state = env.reset()

for i in range(0,10000):
    action = -0.5*state[1:4]-0.5*state[4:7]
    state, reward, done,_ = env.step(action)
    print(np.dot(action,action)/0.09)
    print("step={0} reward={1}".format(i,reward))

states = np.array(env.state_list)
plt.figure(1)
plt.subplot(221)
plt.plot(states[:,0])
plt.subplot(223)
for i in range(1,4):
    plt.plot(states[:,i])

plt.subplot(222)
for i in range(4,7):
    plt.plot(states[:,i])

plt.subplot(224)
for i in range(7,10):
    plt.plot(states[:,i])

plt.show()
