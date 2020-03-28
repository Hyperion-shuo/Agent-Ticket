from Env import Env
# from algorithm.BrainAC import ActorCritic
from algorithm.BrainDQN import BrainDQN
from OrderGenerate import OrderGenerator, readRoute
import numpy as np
import random
from datetime import*
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import os
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def to_state_dqn(his_price, order_price, day):
    his_t = his_price.shape[0]
    s = his_price.copy()
    for i in range(his_t):
        if day + i + 1 <= 86:
            s[i, :day + i + 1] -= order_price
        else:
            s[i, :] -= order_price
    return -s


class TikcetPlay():
    def __init__(self, history_take_off=1, order_num=1):
        self.order = []
        self.routeline = []
        self.allRoute = readRoute("Wang/data/route")
        self.env = Env(self.allRoute, history_take_off=history_take_off, order_num=order_num)
        self.epsilon = 0.


    def transcation_AC(self):
        total_steps = 0  # 记录步数，一天是一步
        profit_list = []  # 记录每局总收益
        profitAdvanced_list = []
        actions = 2  # 行动个数
        brain = ActorCritic(
            n_actions=actions,
            n_features=87,
            LR_A=0.001,
            LR_C=0.01,
            reward_decay=1.,
            prob_clip=0.,
        )
        gameNum = 0 #记录游戏轮数
        ex_steps = 100 #探索衰减的轮数
        epsilon = self.epsilon
        reward_list = [0] #存储每次的收益，来计算baseline
        Loss_list = [] #存储训练过程中的损失值
        wait_list = [] #记录N轮游戏分别等待天数
        gameSplit = 500 #每多少轮游戏画图
        while total_steps < 60000:
            # 初始化游戏
            gameNum += 1
            state = self.env.reset()

            terminal = False
            isExploration = False
            end_date = 0
            # 一局游戏
            baseline = 0
            tao_prob = []
            tao_reward = []
            wait_day = []#记录一局游戏等待哪些天

            while terminal == False:
                today = self.env.getTodayIndex()
                # 当前状态
                state_tf = state[1][0]
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    # end_date = random.randrange(self.env.getTodayIndex(),87,1)
                    end_date = 60

                if isExploration:
                    if today == end_date:
                        action = 1
                        if ex_steps>0:
                            ex_steps -= 1
                    else:
                        action = 0
                else:
                    #action from learning
                    action,p = brain.choose_action(state_tf, today)

                    tao_prob.append(p)

                # 订单字典 历史曲线 reward
                next_state,reward,terminal,_ = self.env.SeparateStep(1, [action])
                today = self.env.getTodayIndex()
                tao_reward.append(reward)
                state_ = next_state[1][0]

                if today >= 0:
                    wait_day.append(today)
                    td_error = brain.criticLearn(state_tf, reward[1], state_)
                    baseline = td_error
                    profitAdvanced_list.append(td_error[0][0])
                    loss = brain.actorLearn(state_tf, action, td_error)
                    # print(loss)
                    Loss_list.append(loss)

                # 保存记录到记忆库
                # print("this is store arg:",state_tf,";", action,";", reward,";", env.getTodayIndex())
                # brain.store_transition(state_tf, action, reward, env.getTodayIndex())
                # print(action)

                total_steps += 1
                if terminal:
                    wait_list.append(wait_day[-1])
                    break
                state = next_state

            # 一局的总收益
            epsilon = self.epsilon*(ex_steps/100)
            print("epsilon:",ex_steps)
            print("TD_Error:",baseline)
            profit = self.env.getTotalReward()
            profit_list.append(profit)
            print("total_steps:",total_steps)
            print("profit_list", profit_list)
            print("profit:", profit, "profitAvg:", np.mean(profit_list))
            print("action-prob:",tao_prob)
            print("Reward:",tao_reward)
            print("wait_day:",wait_day)
            self.writeHistory('./picture/history.txt',
                              epsilon,baseline,total_steps,profit_list,profit,tao_prob,tao_reward,wait_day,gameNum)

            print("########################"+str(gameNum)+"###########################")
            if len(profit_list) >=gameSplit:
                plt.figure()
                plt.plot(profit_list, 'r-')
                plt.savefig('./picture/' + str(gameNum) + 'liner_profit_PG.jpg')
                plt.figure()
                plt.scatter(np.arange(gameSplit), profit_list)
                plt.savefig('./picture/' + str(gameNum) + 'scatter_profit_PG.jpg')
                plt.figure()
                plt.plot(profitAdvanced_list, 'g-')
                plt.savefig('./picture/' + str(gameNum) + 'liner_advanced_PG.jpg')
                plt.figure()
                plt.plot(Loss_list, 'y-')
                plt.savefig('./picture/' + str(gameNum) + 'liner_loss_PG.jpg')
                plt.figure()
                plt.scatter(np.arange(gameSplit), wait_list,c='r')
                plt.savefig('./picture/' + str(gameNum) + 'scatter_waitDay_PG.jpg')
                profit_list.clear()
                wait_list.clear()
            # last_remainder = total_steps % 1000
                # 存储训练过程

    def train_DQN(self, max_game=1000, epsilon=.95):
        # 初始化RL Brain
        actions = 2  # 行动个数
        brain = BrainDQN(actions)
        total_steps = 0
        game_num = 0
        profit_list, day_list = [], []

        while game_num < max_game:
            # 初始化游戏
            game_num += 1
            obs, done = self.env.reset()
            reward = 0
            info = {}
            flag = np.random.rand()

            order_buy_day = []
            if brain.epsilon > flag:
                while not done:
                    today = self.env.getTodayIndex()
                    next_day = today + 1 if today + 1 <= 86 else 86
                    order_num = len(obs["orders"])
                    action = np.zeros(order_num)  # order_num为0时长度为0
                    next_order_buy_day = []
                    # print('len his_order %d, today %d' % (len(obs['his_order']), today))
                    for i in range(order_num):
                        # s = (obs["his_price"]).transpose()[:,:,None]
                        # print(today, obs['his_price'])
                        s = to_state_dqn(obs['his_price'], obs['orders'][i], today)
                        # print(today, s)
                        s = s.transpose()[:,:,None]
                        # print('s shanpe', s.shape)
                        if today == order_buy_day[i]:
                            a = [0, 1]
                            order_buy_day[i] = -1
                        else:
                            a = [1, 0]
                        r = obs["orders"][i] - obs["his_price"][0, today] if a[1] == 1 else 0
                        r_norm = r / 200
                        # 已经对最后一天做了处理
                        # s_ = (self.env.getNextPrice()).transpose()[:,:,None]
                        # print(next_day, self.env.getNextPrice())
                        s_ = to_state_dqn(self.env.getNextPrice(), obs['orders'][i], next_day)
                        # print(next_day, s_)
                        s_ = s_.transpose()[:,:,None]
                        done = True if (today >= 86 or a[1] == 1) else False
                        brain.store_transition(s, a, r_norm, s_, done)
                        # print("today:%d, s:%d, a:%d, r:%d, s_:%d, done: %d" % (today, s[today, 0, 0], a[1], r, s_[next_day, 0, 0], done))
                        action[i] = a[1]
                        if a[1] == 1:
                            day_list.append(today)
                    for i in range(len(order_buy_day)):
                        if order_buy_day[i] != -1:
                            next_order_buy_day.append(order_buy_day[i])
                    order_buy_day = next_order_buy_day
                    if obs["his_order"][-1] != -1 and obs["order_left"] > 0:
                        order_buy_day.append(np.random.choice(np.arange(today + 1, 87, 1)))
                    obs, reward, done, info = self.env.separateStep(1, action)
                    if len(action) > 0:
                        for i in range(len(action)):
                            if action[i] == 1:
                                print("random: day %d, action %d， reward %d" % (today, action[i], reward["reward_buy"]))
                    total_steps += 1
                    if total_steps > 100:
                        brain.trainQNetwork()
                    # if total_steps == 200:
                    #     print(brain.eval_model.summary())
                # print(len(self.env.orders))
            else:
                while not done:
                    today = self.env.getTodayIndex()
                    next_day = today + 1 if today + 1 <= 86 else 86
                    order_num = len(obs["orders"])
                    action = np.zeros(order_num) # order_num为0时长度为0
                    for i in range(order_num):
                        # s = (obs["his_price"]).transpose()[:,:,None]
                        s = to_state_dqn(obs['his_price'], obs['orders'][i], today).transpose()[:, :, None]
                        a = brain.getAction(s, today)
                        r = (obs["orders"][i] - obs["his_price"][0, today]) if a[1]==1 else 0
                        r_norm = r / 200
                        # 已经对最后一天做了处理
                        # s_ = (self.env.getNextPrice()).transpose()[:,:,None]
                        s_ = to_state_dqn(self.env.getNextPrice(), obs['orders'][i], today + 1).transpose()[:, :, None]
                        done = True if (self.env.getTodayIndex() >= 86 or a[1] == 1) else False
                        brain.store_transition(s, a, r_norm, s_, done)
                        # print("today: %d,s:%d, a:%d, r:%d, s_:%d, done: %d" % (today, s[today,0,0], a[1], r, s_[next_day,0,0], done))
                        action[i] = a[1]
                        if a[1] == 1:
                            day_list.append(today)
                    obs, reward, done, info = self.env.separateStep(1, action)
                    if len(action) > 0:
                        for i in range(len(action)):
                            if action[i] == 1:
                                print("agent: day %d, action %d， reward %d" % (today, action[i], reward["reward_buy"]))
                    total_steps += 1
                    if total_steps > 100:
                        brain.trainQNetwork()
            profit_list.append(self.env.getTotalReward())
            if game_num % 10 == 0:
            #     plt.plot()
            #     plt.xlabel("game")
            #     plt.ylabel("reward")
            #     plt.savefig('shen/picture/' + "reward" + "_game_" + str(game_num) + '_.png')
            #     plt.cla()
                print("avg_profit: %f, avg_day %f" % (np.average(profit_list), np.average(day_list)))



    def writeHistory(self, filename, epsilon, baseline, total_steps, profit_list, profit, tao_prob, tao_reward,
                     wait_day, gameNum):
        f = open(filename, 'a')
        f.write("epsilon:" + str(epsilon) + "\n")
        f.write("Baseline:" + str(baseline) + "\n")
        f.write("total_steps:" + str(total_steps) + "\n")
        f.write("profit_list" + str(profit_list) + "\n")
        f.write("profit:" + str(profit) + "profitAvg:" + str(np.mean(profit_list)) + "\n")
        f.write("action-prob:" + str(tao_prob) + "\n")
        f.write("Reward:" + str(tao_reward) + "\n")
        f.write("wait_day:" + str(wait_day) + "\n")
        f.write("########################" + str(gameNum) + "###########################\n")
        f.flush()



if __name__ == "__main__":
    # P = TikcetPlay()
    # P.transcation_AC()
    P = TikcetPlay(history_take_off=2, order_num=10)
    P.train_DQN(max_game=600)

