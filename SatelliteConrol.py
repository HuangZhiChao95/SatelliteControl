import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from env.EnvBlock import envblock
from env.SatelliteEnv import SatelliteEnv
from policygradient.PolicyGradient import PolicyGradient
import argparse
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Semaphore, Queue
from math import exp
import time

parser = argparse.ArgumentParser()
parser.add_argument("--method", help="Method for Reinforcement Learning, currently support PolicyGradient",
                    default="PolicyGradient")
parser.add_argument("--iteration", type=int, default=10000)
parser.add_argument("--tspan", type=float, default=0.5)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--processnum", type=int, default=4)
parser.add_argument("savename")

args = parser.parse_args()

iteration = args.iteration
method = args.method
batchsize = args.batchsize
processnum = args.processnum
config = tf.ConfigProto()
config.log_device_placement = False
config.gpu_options.allow_growth = True

states = list()
penv = {
    "tspan": args.tspan,
    "theta": (2*np.random.rand((5000,3))-1)*np.diag([0.5, 0.5, 0.3]),
    "wb":(2*np.random.rand((5000,3))-1)*np.diag([0.001,0.001,0.001])
}
p_testenv = {
    "tspan": args.tspan,
    "theta": np.array([0.5, 0.5, 0.3]),
    "wb":np.ones(3)*0.02*np.pi/180
}
processes = list()
queues = list()
lockenvs= list()
lockmains = list()

for i in range(processnum):
    lockenv = Semaphore(0)
    lockmain = Semaphore(0)
    q = Queue()
    process = Process(target=envblock, args=(batchsize, penv, lockenv, lockmain, q))

    processes.append(process)
    lockenvs.append(lockenv)
    lockmains.append(lockmain)
    queues.append(q)

    process.start()


if method == "PolicyGradient":
    agent = PolicyGradient(init_std=1e-4)
    reward_list = list()
    rewards = np.zeros(processnum * batchsize, dtype=np.float32)
    states = np.zeros([processnum * batchsize, 7], dtype=np.float32)
    dones = np.ndarray(processnum * batchsize, dtype=np.bool)
    saver = tf.train.Saver()
    lr_rate = 1e-3
    summary_writer = tf.summary.FileWriter('./log')
    test_env = SatelliteEnv(p_testenv)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(0, iteration):

            #save record
            if i % 100 == 0:
                saver.save(sess, "./model/{0}.ckpt".format(args.savename), global_step=i)
                env = test_env
                state = env.reset()
                state = state[np.newaxis, :]

                for k in range(0, int(2000 / args.tspan)):
                    action = agent.predict(state, sess, std=0.0)
                    state, reward, done, __ = env.step(action)
                    state = state[np.newaxis, :]
                    reward_list.append(reward / exp(k * args.tspan / 500))

                result = {
                    "state": np.array(agent.state_list),
                    "action": np.array(agent.action_list),
                    "reward": np.array(reward_list)
                }

                if not os.path.exists("record"):
                    os.mkdir("record")

                np.save("./record/{1}_{0}".format(i, args.savename), np.array(result))
                reward_list.clear()
                agent.clearMem()

            #reset
            for j in range(processnum):
                queues[j].put("reset")
                lockenvs[j].release()

            for j in range(processnum):
                lockmains[j].acquire()

            for j in range(processnum):
                states[batchsize*j:(j+1)*batchsize,:] = queues[j].get()

            k = 0
            Done = False

            while not Done:
                actions = agent.predict(states, sess)
                flag = True
                for j in range(processnum):
                    queues[j].put(actions[batchsize*j:(j+1)*batchsize,:])
                    lockenvs[j].release()

                for j in range(processnum):
                    lockmains[j].acquire()

                for j in range(processnum):
                    state_block, reward_block, done_block = queues[j].get()
                    states[batchsize*j:(j+1)*batchsize, :] = state_block
                    rewards[batchsize*j:(j+1)*batchsize] =reward_block
                    dones[batchsize*j:(j+1)*batchsize] = done_block

                Done = np.all(dones)
                reward_list.append(rewards.copy())

                if k % 100 == 0 and i % 10 == 0:
                    print("iteration={0} step={1} reward={2} std={3}".format(i, k, np.mean(
                        rewards / exp(k * args.tspan / 500)), np.std(rewards / exp(k * args.tspan / 500))))
                k = k + 1

            target, loss, summary_str = agent.update(reward_list, lr_rate, sess)

            if i % 500 == 0:
                lr_rate = lr_rate / 2

            summary_writer.add_summary(summary_str, i)
            print("iteration{0} loss={3}, target={1}, step={2}".format(i, np.mean(target), k, loss))
            reward_list.clear()
