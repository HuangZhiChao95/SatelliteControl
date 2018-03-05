import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("recordpath")
args = parser.parse_args()
recordpath = "record/"+args.recordpath
if not recordpath.endswith(".npy"):
    recordpath = recordpath+".npy"
record = np.load(recordpath).item()
states = record["state"]
actions = record["action"]
rewards = record["reward"]

plt.figure(1)
plt.subplot(231)
plt.plot(states[:, 0, 0])
plt.title("q0")
# plt.xlabel(u"time")
# plt.ylabel(u"q0")

plt.subplot(232)
plt.title("q1-q3")
for i in range(1,4):
    plt.plot(states[:, 0, i])

plt.subplot(233)
plt.title("v1-v3")
for i in range(4,7):
    plt.plot(states[:, 0, i])

plt.subplot(234)
plt.title("action1-action3")
for i in range(0,3):
    plt.plot(actions[:,i])

plt.subplot(235)
plt.title("reward")
plt.plot(rewards)

plt.show()