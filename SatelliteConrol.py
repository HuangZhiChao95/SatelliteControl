import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from env.SatelliteEnv import SatelliteEnv
from policygradient.PolicyGradient import PolicyGradient
import argparse
import numpy as np
import tensorflow as tf
from math import exp

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
config.gpu_options.allow_growth = True

states = list()
envs = list()
for i in range(batchsize):
    env = SatelliteEnv({"tspan": args.tspan, "theta": np.array([0.5, 0.5, 0.3])})
    envs.append(env)



if method=="PolicyGradient":
    agent = PolicyGradient(init_std=1e-4)
    reward_list = list()
    rewards = np.zeros(batchsize, dtype=np.float32)
    saver = tf.train.Saver()
    lr_rate = 1e-3
    with tf.device('/gpu:0'):
        summary_writer = tf.summary.FileWriter('./log')
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(0, iteration):
                Done = False
                states = list()

                if i % 100 == 0:
                    saver.save(sess, "./model/fixedstd.ckpt", global_step=i)
                    env = envs[0]
                    state = env.reset()
                    state = state[np.newaxis, :]

                    for k in range(0,int(2000/args.tspan)):
                        action = agent.predict(state, sess, std=0.0)
                        state, reward, done, __ = env.step(action)
                        state = state[np.newaxis, :]
                        reward_list.append(reward/exp(k*args.tspan/500))

                    result = {
                                "state":np.array(agent.state_list),
                                "action": np.array(agent.action_list),
                                "reward": np.array(reward_list)
                              }

                    if os.path.exists("record"):
                        os.mkdir("record")

                    np.save("./record/fixedstd_{0}".format(i), np.array(result))
                    reward_list.clear()
                    agent.clearMem()

                k=0
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

                    if k % 100 ==0 and i % 10 ==0:
                        print("iteration={0}, step={1} reward={2} std={3}".format(i, k, np.mean(rewards/exp(k*args.tspan/500)), np.std(rewards/exp(k*args.tspan/500))))
                    k = k+1

                target, loss, summary_str = agent.update(reward_list, lr_rate, sess)

                if i % 500 == 0:
                    lr_rate = lr_rate/2

                summary_writer.add_summary(summary_str, i)
                print("iteration{0}, loss={3}, target={1}, step={2}".format(i, np.mean(target), k, loss))
                reward_list.clear()





