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
        self.history_take_off = history_take_off


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
            RNN_num = self.history_take_off
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
                state_tf = state['his_price']
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    end_date = random.randrange(today+1,87,1)
                    # end_date = 60

                order_num = len(state["orders"])
                action = []
                p = []
                if order_num > 0:
                    for i in range(order_num):
                        if isExploration:
                            if today >= end_date:
                                action.append(1)
                                if ex_steps > 0:
                                    ex_steps -= 1
                            else:
                                action.append(0)
                            p.append([-1, -1])
                        else:
                            # action from learning
                            action_, p_ = brain.choose_action(state_tf,today+1)
                            action.append(action_)
                            p.append(p_)

                        tao_prob.append(p)

                # 订单字典 历史曲线 reward
                next_state,reward,terminal,_ = self.env.separateStep(1, action)
                today = self.env.getTodayIndex()
                tao_reward.append(reward)
                state_ = next_state['his_price']

                if today >= 0 and len(action)>0:
                    wait_day.append(today)
                    td_error = brain.criticLearn(state_tf, reward['reward_buy'], state_, today)
                    baseline = td_error
                    profitAdvanced_list.append(td_error[0][0])
                    # print(state_)
                    # print(action)
                    # print(td_error)
                    loss = brain.actorLearn(state_tf, action[0], td_error, today)
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
            GAMMA=.9,
            TAU=0.01,
            per_memory_size=2000,
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
            td_error = 0
            tao_prob = []
            tao_reward = []
            wait_day = []#记录一局游戏等待哪些天

            while terminal == False:
                today = self.env.getTodayIndex()
                # 当前状态
                # print(state["his_price"])
                state_tf = state['his_price'][0]
                state_tf = self.statePreprocess(state_tf)
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    # end_date = random.randrange(self.env.getTodayIndex(),87,1)
                    end_date = 60

                order_num = len(state["orders"])
                action = []
                p = []
                if order_num>0:
                    for i in range(order_num):
                        if isExploration:
                            if today >= end_date:
                                action.append(1)
                                if ex_steps>0:
                                    ex_steps -= 1
                            else:
                                action.append(0)
                            p.append([-1,-1])
                        else:
                            #action from learning
                            action_,p_ = brain.choose_action(state_tf)
                            action.append(action_)
                            p.append(p_)

                        tao_prob.append(p)

                # 订单字典 历史曲线 reward
                next_state,reward,terminal,_ = self.env.separateStep(1, action)
                today = self.env.getTodayIndex()
                tao_reward.append(reward)
                state_ = next_state['his_price'][0]

                if today >= 0 and len(action)>0:
                    # print("today_",today)
                    total_steps += 1
                    wait_day.append(today)
                    state_ = self.statePreprocess(state_)
                    brain.store_transition(state_tf, action, [reward['reward_buy']], state_)
                    # print(action)

                if total_steps > brain.per_memory_size :
                    # print(b_s_)
                    td_error,loss = brain.learn()
                    profitAdvanced_list.append(td_error)
                    Loss_list.append(loss)
                    # print("*"*10+"start learn")



                if terminal:
                    wait_list.append(wait_day[-1])
                    # print("wait_list::",wait_list)
                    break
                state = next_state

            # 一局的总收益
            epsilon = self.epsilon * (ex_steps / 500)
            print("epsilon:", epsilon)
            print("TD_Error:", td_error)
            profit = self.env.getTotalReward()
            profit_list.append(profit)
            print("total_steps:", total_steps)
            print("profit_list", profit_list)
            print("profit:", profit, "profitAvg:", np.mean(profit_list))
            print("action-prob:", tao_prob)
            print("Reward:", tao_reward)
            print("wait_day:", wait_day)
            self.writeHistory('./picture/history.txt',
                              epsilon, td_error, total_steps, profit_list, profit, tao_prob, tao_reward, wait_day,
                              gameNum)

            print("########################" + str(gameNum) + "###########################")
            if len(profit_list) >= gameSplit:
                plt.figure()
                plt.plot(profit_list, 'r-')
                plt.savefig('./picture/' + str(gameNum) + 'liner_profit_DDPG.jpg')
                plt.figure()
                plt.scatter(np.arange(gameSplit), profit_list)
                plt.savefig('./picture/' + str(gameNum) + 'scatter_profit_DDPG.jpg')
                plt.figure()
                # plt.plot(profitAdvanced_list, 'g-')
                # plt.savefig('./picture/' + str(gameNum) + 'liner_advanced_DDPG.jpg')
                reslut_list = [item for sublist in profitAdvanced_list for item in sublist]
                plt.scatter(np.arange(len(reslut_list)), reslut_list,c='g')
                plt.savefig('./picture/' + str(gameNum) + 'scatter_TDERROR_DDPG.jpg')
                plt.figure()
                plt.plot(Loss_list, 'y-')
                plt.savefig('./picture/' + str(gameNum) + 'liner_loss_DDPG.jpg')
                plt.figure()
                plt.scatter(np.arange(gameSplit), wait_list, c='r')
                plt.savefig('./picture/' + str(gameNum) + 'scatter_waitDay_DDPG.jpg')
            if len(profit_list) >= 500:
                profit_list.clear()
                wait_list.clear()
                # last_remainder = total_steps % 1000
            # 存储训练过程
    #输入数据预处理
    def statePreprocess(self,state):
        exist = np.array([n for n in state if n>0])
        exist -= 1877.368
        exist /= 256.61
        exist = exist.tolist()
        while len(exist) < 87:
            exist.append(0)
        # print("STATE:", self.state,"EXIST:",exist)
        return exist

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
    P = TikcetPlay(history_take_off=5, order_num=1)
    # P.transcation_DDPG()
    P.transcation_AC()