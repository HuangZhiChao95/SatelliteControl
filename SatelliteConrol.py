import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from env.SatelliteEnv import SatelliteEnv
from policygradient.PolicyGradient import PolicyGradient
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--method", help="Method for Reinforcement Learning, currently support PolicyGradient", default="PolicyGradient")
parser.add_argument("--iteration", type=int, default=10000)
parser.add_argument("--tspan", type=float, default=0.5)
parser.add_argument("--batchsize", type=int, default=64)

args = parser.parse_args()

iteration = args.iteration
method = args.method
batchsize = args.batchsize
config = tf.ConfigProto()
config.log_device_placement = True

states = list()
envs = list()
for i in range(batchsize):
    env = SatelliteEnv({"tspan": args.tspan})
    envs.append(env)



if method=="PolicyGradient":
    agent = PolicyGradient(init_std=1e-4)
    reward_list = list()
    rewards = np.zeros(batchsize, dtype=np.float32)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, iteration):
            Done = False
            k=0
            states = list()
            for env in envs:
                state = env.reset()
                states.append(state)
            states = np.array(states, dtype=np.float32)

            while not Done:
                actions = agent.predict(states, sess)
                flag = True
                for j in range(len(actions)):
                    state, reward, done, __ = envs[j].step(actions[j])
                    states[j, :] = state
                    rewards[j] = reward
                    if not done:
                        flag = False
                Done = flag
                reward_list.append(rewards.copy())
                k = k+1
                if k % 100 ==0 and k>0:
                    print("iteration={0}, step={1} reward={2}".format(i, k, np.mean(rewards)))

            target = agent.update(reward_list, 1e-4, sess)
            print("iteration{0}, target={1}, step={2}".format(i, np.mean(np.array(reward_list)), k))
            reward_list.clear()





