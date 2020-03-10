'''
为了不与其他人调试不同算法的时候，频繁修改同一个py文件，因此自己新赠一个转移文件，主要包含主函数，数据与处理，结果输出等内容
Wang
2020-03-09
'''
from Env import Env
from algorithm.BrainAC import ActorCritic
# from algorithm.BrainDQN import BrainDQN
from algorithm.BrainDDPG import DDPG
from OrderGenerate import OrderGenerator, readRoute
import numpy as np
import random
from  datetime import*
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class TikcetPlay():
    def __init__(self, history_take_off=1, order_num=1):
        self.order = []
        self.routeline = []
        self.allRoute = readRoute("./wang/data/route")
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

    def transcation_DDPG(self):
        BATCH_SIZE = 32
        total_steps = 0  # 记录步数，一天是一步
        profit_list = []  # 记录每局总收益
        profitAdvanced_list = []
        actions = 2  # 行动个数
        s_dim = 87
        a_dim = 1
        brain = DDPG(
            a_dim=a_dim,
            s_dim=s_dim,
            a_bound=1.,
            LR_A=0.001,
            LR_C=0.001,
            GAMMA=.99,
            TAU=0.01,
            # replacement=REPLACEMENT,
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
            state,_ = self.env.reset()

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
                # print(state["his_price"])
                state_tf = state['his_price'][0]
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    # end_date = random.randrange(self.env.getTodayIndex(),87,1)
                    end_date = 60

                order_num = len(state["orders"])
                action = np.zeros(order_num)
                p = []
                if order_num>0:
                    for i in range(order_num):
                        if isExploration:
                            if today >= end_date:
                                action[i] = 1
                                if ex_steps>0:
                                    ex_steps -= 1
                            else:
                                action[i] = 0
                            p.append([-1,-1])
                        else:
                            #action from learning
                            action[i],p_ = brain.choose_action(state_tf)
                            p.append(p_)

                        tao_prob.append(p)

                # 订单字典 历史曲线 reward
                next_state,reward,terminal,_ = self.env.separateStep(1, action)
                today = self.env.getTodayIndex()
                tao_reward.append(reward)
                state_ = next_state['his_price'][0]

                if today >= 0:
                    # print("today_",today)
                    wait_day.append(today)
                    brain.store_transition(state_tf, action, reward, state_)

                if brain.per_memory.tree.data_pointer > brain.per_memory_size :
                    # print(b_s_)
                    brain.learn()

                total_steps += 1
                if terminal:
                    wait_list.append(wait_day[-1])
                    # print("wait_list::",wait_list)
                    break
                state = next_state

            # 一局的总收益
            epsilon = self.epsilon * (ex_steps / 500)
            print("epsilon:", epsilon)
            print("TD_Error:", baseline)
            profit = self.env.getTotalReward()
            profit_list.append(profit)
            print("total_steps:", total_steps)
            print("profit_list", profit_list)
            print("profit:", profit, "profitAvg:", np.mean(profit_list))
            print("action-prob:", tao_prob)
            print("Reward:", tao_reward)
            print("wait_day:", wait_day)
            self.writeHistory('./picture/history.txt',
                              epsilon, baseline, total_steps, profit_list, profit, tao_prob, tao_reward, wait_day,
                              gameNum)

            print("########################" + str(gameNum) + "###########################")
            if len(profit_list) >= gameSplit:
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
                plt.scatter(np.arange(gameSplit), wait_list, c='r')
                plt.savefig('./picture/' + str(gameNum) + 'scatter_waitDay_PG.jpg')
            if len(profit_list) >= 500:
                profit_list.clear()
                wait_list.clear()
                # last_remainder = total_steps % 1000
            # 存储训练过程

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
    P = TikcetPlay()
    P.transcation_DDPG()

