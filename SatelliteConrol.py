import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from env.EnvBlock import envblock
from env.SatelliteEnv import SatelliteEnv
from policygradient.PolicyGradient import PolicyGradient
from DDPG.DDPG import DDPG
from LQR.Model import Model

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
parser.add_argument("--batchsize_sample", type=int, default=256)
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
    "theta": np.matmul(2*np.random.rand(5000,3)-1, np.diag([0.5, 0.5, 0.3])),
    "wb":np.matmul(2*np.random.rand(5000,3)-1, np.diag([0.001,0.001,0.001]))
}
'''
penv = {
    "tspan": args.tspan,
    "theta": np.array([[0.6, 0.5, 0.4]]),
    "wb":np.ones((1, 3))*0.02*np.pi/180
}'''
p_testenv = {
    "tspan": args.tspan,
    "theta": np.array([[0.6, 0.5, 0.4]]),
    "wb":np.ones((1, 3))*0.02*np.pi/180
}
processes = list()
queues = list()
lockenvs= list()
lockmains = list()

for i in range(processnum):
    lockenv = Semaphore(0)
    lockmain = Semaphore(0)
    q = Queue()
    process = Process(target=envblock, args=(batchsize, penv, lockenv, lockmain, q, 7))

    processes.append(process)
    lockenvs.append(lockenv)
    lockmains.append(lockmain)
    queues.append(q)

    process.start()


if method == "PolicyGradient":
    agent = PolicyGradient(init_std=1e-2, input_dim=7)


    reward_list = list()
    rewards = np.zeros(processnum * batchsize, dtype=np.float32)
    states = np.zeros([processnum * batchsize, 7], dtype=np.float32)
    dones = np.ndarray(processnum * batchsize, dtype=np.bool)
    saver = tf.train.Saver()
    lr_rate = 5e-3
    summary_writer = tf.summary.FileWriter('./log')
    test_env = SatelliteEnv(p_testenv)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        '''
        pd_lr_rate = 40.0
        for i in range(100000):
            q1 = np.random.rand(batchsize, 3) * 0.3#np.ones([batchsize,3])*0.3#
            q0 = np.sqrt(1 - np.sum(np.square(q1), axis=1))
            q0 = q0[:, np.newaxis]
            w = np.random.rand(batchsize, 3) * 0.01 #np.ones([batchsize,3])*0.01#
            sq = (2 * np.random.rand(batchsize, 4) - 1) * 10
            pd_actions = -0.5 * q1 - 0.5 * w - 0.001 * sq[:, 1:]
            pd_states = np.concatenate((q0, q1, w, sq), axis=1)
            loss_pd = agent.imitation_learn(states=pd_states, actions=pd_actions, lr_rate=pd_lr_rate, sess=sess)
            if i % 10000==0:
                pd_lr_rate = pd_lr_rate/2
            if i % 100 ==0:
                print("imitate pd, step={0}, loss={1}".format(i, loss_pd))'''

        for i in range(0, iteration):

            #save record
            if i % 5 == 0:
                saver.save(sess, "./model/{0}.ckpt".format(args.savename), global_step=i)
                env = test_env
                state = env.reset()
                state = state[np.newaxis, :]

                for k in range(0, int(2000 / args.tspan)):
                    action = agent.predict(state, sess, std=0.0)
                    state, reward, done, __ = env.step(action)
                    state = state[np.newaxis, :]
                    reward_list.append(reward) #/ exp(k * args.tspan / 500))

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
                    #print("iteration={0} step={1} reward={2} std={3}".format(i, k, np.mean(
                    #    rewards / exp(k * args.tspan / 500)), np.std(rewards / exp(k * args.tspan / 500))))
                    print("iteration={0} step={1} reward={2} std={3}".format(i, k, np.mean(rewards), np.std(rewards)))
                k = k + 1

            target, loss, summary_str = agent.update(reward_list, lr_rate, sess)

            if i % 500 == 0:
                lr_rate = lr_rate / 2

            summary_writer.add_summary(summary_str, i)
            print("iteration{0} loss={3}, target={1}, step={2}".format(i, np.mean(target), k, loss))
            reward_list.clear()

if method == "DDPG":
    agent = DDPG(tau=0.1, replaysize=50000, batch_size=256, state_dim=7)
    rewards = np.zeros(processnum * batchsize, dtype=np.float32)
    states = np.zeros([processnum * batchsize, 7], dtype=np.float32)
    next_states = np.zeros([processnum * batchsize, 7], dtype=np.float32)
    dones = np.ndarray(processnum * batchsize, dtype=np.bool)
    saver = tf.train.Saver()
    lr_rate = 1e-5
    test_env = SatelliteEnv(penv)
    with tf.Session(config=config) as sess:
        summary_op = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log',sess.graph)
        
        for i in range(0, iteration):

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)

            # save record
            if i % 20 == 0:
                print("record")
                saver.save(sess, "./model/{0}.ckpt".format(args.savename), global_step=i)
                env = test_env
                action_list = []
                state_list = []
                reward_list = []
                state = env.reset()
                state = state[np.newaxis, :]

                for k in range(0, int(1500 / args.tspan)):
                    action = agent.predict(state, sess).flat
                    #print(state.shape)
                    #action = -0.5*state[:,1:4]-0.5*state[:,4:7]
                    #action = action.float
                    state, reward, done, __ = env.step(action)
                    state_list.append(state.copy())
                    state = state[np.newaxis, :]
                    #print(reward)
                    action_list.append(action.copy())
                    reward_list.append(reward.copy()) #/ exp(k * args.tspan / 500))

                result = {
                    "state": np.array(state_list),
                    "action": np.array(action_list),
                    "reward": np.array(reward_list)
                }

                if not os.path.exists("record"):
                    os.mkdir("record")

                np.save("./record/{1}_{0}".format(i, args.savename), np.array(result))
                reward_list.clear()

            # reset
            for j in range(processnum):
                queues[j].put("reset")
                lockenvs[j].release()

            for j in range(processnum):
                lockmains[j].acquire()

            for j in range(processnum):
                states[batchsize * j:(j + 1) * batchsize, :] = queues[j].get()

            Done = False
            k = 0
            while not Done:
                if i==10 and k % 10==0:
                    print("record")
                    saver.save(sess, "./model/{0}.ckpt".format(args.savename), global_step=i)
                    env = test_env
                    action_list = []
                    state_list = []
                    reward_list = []
                    state = env.reset()
                    state = state[np.newaxis, :]

                    for j in range(0, int(1500 / args.tspan)):
                        action = agent.predict(state, sess).flat
                        #print(state.shape)
                        #action = -0.5*state[:,1:4]-0.5*state[:,4:7]
                        #action = action.float
                        state, reward, done, __ = env.step(action)
                        state_list.append(state.copy())
                        state = state[np.newaxis, :]
                        #print(reward)
                        action_list.append(action.copy())
                        reward_list.append(reward.copy()) #/ exp(k * args.tspan / 500))

                    result = {
                        "state": np.array(state_list),
                        "action": np.array(action_list),
                        "reward": np.array(reward_list)
                    }                    
                    if not os.path.exists("record"):
                        os.mkdir("record")
                    print("./record/{1}_{0}_{2}".format(i, args.savename, k))
                    np.save("./record/{1}_{0}_{2}".format(i, args.savename, k), np.array(result))
                    reward_list.clear()                
                    
                if i<10:
                    actions = - 0.5 * states[:,1:4]-0.5*states[:,4:7]
                else:
                    actions = agent.predict_noise(states, sess)
                flag = True
                for j in range(processnum):
                    queues[j].put(actions[batchsize * j:(j + 1) * batchsize, :])
                    lockenvs[j].release()

                for j in range(processnum):
                    lockmains[j].acquire()

                for j in range(processnum):
                    state_block, reward_block, done_block = queues[j].get()
                    next_states[batchsize * j:(j + 1) * batchsize, :] = state_block
                    rewards[batchsize * j:(j + 1) * batchsize] = reward_block
                    dones[batchsize * j:(j + 1) * batchsize] = done_block
                Done = np.all(dones)                  
                agent.store_sample(states.copy(), next_states.copy(), actions.copy(), rewards.copy())
                states = next_states
                if i<10:
                    loss, pd_loss, fitQ, r,Q_all = agent.update(lr_rate=lr_rate, sess=sess, imitate_pd=True)
                    if k % 100 == 0:
                        print("iteration={0} step={5} fit_loss={1} pd_loss={2} fitQ={3} r={4}".format(i, loss, pd_loss, np.mean(fitQ), np.mean(r), k))
                else:
                    loss, Q, fitQ, r,Q_all = agent.update(lr_rate=lr_rate, sess=sess)
                    if k % 100 == 0:
                        print("iteration={0} step={5} fit_loss={1} Q_value={2} fitQ={3} r={4}".format(i, loss, np.mean(Q_all), np.mean(fitQ), np.mean(r), k))
                    #print("diff={0}".format(np.abs(Q_all-fitQ)))
                    #print(rewards)
                k = k + 1
            print(k)
            if i % 500 == 0:
                lr_rate = lr_rate / 2

if method=="LQR":
    model = Model(init_std=1e-2)
    model.run(1e-3,100000)