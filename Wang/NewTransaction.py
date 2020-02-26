from Environment import Env
from BrainPG import PolicyGradient
from BrainAC import ActorCritic
from BrainDDPG import DDPG
import numpy as np
import random
from  datetime import*
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import os
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class TikcetPlay():
    def __init__(self):
        self.order = []
        self.routeline = []
        # self.allRoute = self.readRoute("/home/pycharm_workspace/Ticket_bak/data/route")
        # self.allOrder = self.readOrder("/home/pycharm_workspace/Ticket_bak/data/order")
        self.allRoute = self.readRoute("./data/route")
        self.allOrder = self.readOrder("./data/order")
        self.epsilon = 0.9
        # print(self.allRoute[21])
        # print(self.allOrder[0])

    def transcate_PG(self):
        total_steps = 0  # 记录步数，一天是一步
        profit_list = []  # 记录每局总收益
        profitAdvanced_list = []
        actions = 2  # 行动个数
        brain = PolicyGradient(
            n_actions=2,
            n_features=87,
            learning_rate=0.1,
            reward_decay=1,
        )
        gameNum = 0 #记录游戏轮数
        ex_steps = 500 #探索衰减的轮数
        epsilon = self.epsilon
        last_remainder  = 0
        reward_list = [0] #存储每次的收益，来计算baseline
        Loss_list = [] #存储训练过程中的损失值
        wait_list = [] #记录等待天数
        gameSplit = 500 #每多少轮游戏画图
        while total_steps < 60000:
            # 初始化游戏
            # routeId = random.randrange(0, 49, 1)
            routeId = 21
            self.routeline = self.allRoute[routeId]
            # print(self.routeline)
            env = Env(self.routeline)
            gameNum += 1
            # state = env.getState()  # 以state[0]、state[1]方式访问
            today = env.getToday()

            terminal = False
            order_accepted = False
            isExploration = False
            create_date = 1
            end_date = 0
            stay_num = 0
            # 一局游戏
            # print("GAME#:",gameNum)
            baseline = 0
            tao_prob = []
            tao_reward = 0
            wait_day = []

            while today < self.routeline[-1] and terminal == False:
                # 有新订单产生 (当订单数已满10个时，此处不会收到新订单)
                if order_accepted == False:
                    self.orderSelect(self.routeline,60)
                    # print(self.order)
                    env.setOrder(self.order)
                    order_accepted = True
                    # print(self.order[1])
                # 遍历self.orders(即state[0])字典，对每一个订单操作
                state = env.getState()

                # 当前状态
                state_tf = np.mat(state)
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    end_date = random.randrange(env.getTodayIndex(),87,1)
                    # end_date = 60

                if isExploration:
                    if env.getTodayIndex() == end_date:
                        action_model = 1
                        if ex_steps>0:
                            ex_steps -= 1
                    else:
                        action_model = 0
                else:
                    #action from learning
                    action_model,p = brain.choose_action(state_tf, env.getTodayIndex())
                    tao_prob.append(p)
                if action_model == 0:
                    action_finishOrder = [1, 0]
                else:
                    action_finishOrder = [0, 1]

                # 订单字典 历史曲线 reward

                reward = env.getReward(action_model)

                # 订单完成或者到最后一天
                terminal = env.isTerminal(action_model)
                if terminal:
                    tmp = reward
                    baseline = np.mean(reward_list)
                    profitAdvanced_list.append(baseline)
                    reward -= baseline
                    reward_list.append(tmp)
                    # print("END_REWARD:",reward,",reward_list:",reward_list)
                # 保存记录到记忆库
                # print("this is store arg:",state_tf,";", action_model,";", reward,";", env.getTodayIndex())
                brain.store_transition(state_tf, action_model, reward, env.getTodayIndex())
                # print(action_model)

                total_steps += 1
                if terminal:
                    loss,wait_day,tao_reward = brain.learn()
                    Loss_list.append(loss)
                    wait_list.append(wait_day[-1])
                    break

                # step 过一天加一
                env.nextStep()


            # 一局的总收益
            epsilon = self.epsilon*(ex_steps/500)
            print("epsilon:",epsilon)
            print("Baseline:",baseline)
            profit = env.getTotalReward()
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

    def transcate_AC(self):
        total_steps = 0  # 记录步数，一天是一步
        profit_list = []  # 记录每局总收益
        profitAdvanced_list = []
        actions = 2  # 行动个数
        brain = ActorCritic(
            n_actions=2,
            n_features=87,
            LR_A=0.001,
            LR_C=0.01,
            reward_decay=1.,
            prob_clip=0.,
        )
        gameNum = 0 #记录游戏轮数
        ex_steps = 500 #探索衰减的轮数
        epsilon = self.epsilon
        last_remainder  = 0
        reward_list = [0] #存储每次的收益，来计算baseline
        Loss_list = [] #存储训练过程中的损失值
        wait_list = [] #记录N轮游戏分别等待天数
        gameSplit = 500 #每多少轮游戏画图
        while total_steps < 60000:
            # 初始化游戏
            # routeId = random.randrange(0, 49, 1)
            routeId = 21
            self.routeline = self.allRoute[routeId]
            # print(self.routeline)
            env = Env(self.routeline)
            gameNum += 1
            # state = env.getState()  # 以state[0]、state[1]方式访问
            today = env.getToday()

            terminal = False
            order_accepted = False
            isExploration = False
            create_date = 1
            end_date = 0
            stay_num = 0
            # 一局游戏
            # print("GAME#:",gameNum)
            baseline = 0
            tao_prob = []
            tao_reward = []
            wait_day = []#记录一局游戏等待哪些天

            while today < self.routeline[-1] and terminal == False:
                # 有新订单产生 (当订单数已满10个时，此处不会收到新订单)
                if order_accepted == False:
                    self.orderSelect(self.routeline,60)
                    # print(self.order)
                    env.setOrder(self.order)
                    order_accepted = True
                # 遍历self.orders(即state[0])字典，对每一个订单操作
                state = env.getState()

                # 当前状态
                state_tf = np.mat(state)
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    end_date = random.randrange(env.getTodayIndex(),87,1)
                    # end_date = 60

                if isExploration:
                    if env.getTodayIndex() == end_date:
                        action_model = 1
                        if ex_steps>0:
                            ex_steps -= 1
                    else:
                        action_model = 0
                else:
                    #action from learning
                    action_model,p = brain.choose_action(state_tf, env.getTodayIndex())
                    tao_prob.append(p)
                if action_model == 0:
                    action_finishOrder = [1, 0]
                else:
                    action_finishOrder = [0, 1]

                wait_day.append(env.getTodayIndex())
                # 订单字典 历史曲线 reward

                reward = env.getReward(action_model)
                tao_reward.append(reward)
                # 订单完成或者到最后一天
                terminal = env.isTerminal(action_model)
                state_ = env.getNextState(action_model)
                # print(state_tf)
                # print(state_)
                td_error = brain.criticLearn(state_tf, reward, state_)
                baseline = td_error
                profitAdvanced_list.append(td_error[0][0])
                loss = brain.actorLearn(state_tf, action_model, td_error)
                # print(loss)
                Loss_list.append(loss)
                # 保存记录到记忆库
                # print("this is store arg:",state_tf,";", action_model,";", reward,";", env.getTodayIndex())
                # brain.store_transition(state_tf, action_model, reward, env.getTodayIndex())
                # print(action_model)

                total_steps += 1
                if terminal:
                    wait_list.append(wait_day[-1])
                    break

                # step 过一天加一
                env.nextStep()


            # 一局的总收益
            # epsilon = self.epsilon*(ex_steps/500)
            print("epsilon:",epsilon)
            print("TD_Error:",baseline)
            profit = env.getTotalReward()
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

    def transcate_DDPG(self):
        BATCH_SIZE = 32
        total_steps = 0  # 记录步数，一天是一步
        profit_list = []  # 记录每局总收益
        profitAdvanced_list = []
        actions = 2  # 行动个数
        s_dim = 87
        a_dim = 1
        brain = DDPG(
            a_dim = a_dim,
            s_dim = s_dim,
            a_bound = 1.,
            LR_A = 0.001,
            LR_C = 0.001,
            GAMMA = .99,
            TAU = 0.01,
            # replacement=REPLACEMENT,
        )
        gameNum = 0 #记录游戏轮数
        ex_steps = 500 #探索衰减的轮数
        epsilon = self.epsilon
        last_remainder  = 0
        reward_list = [0] #存储每次的收益，来计算baseline
        Loss_list = [] #存储训练过程中的损失值
        wait_list = [] #记录N轮游戏分别等待天数
        gameSplit = 5000 #每多少轮游戏画图
        while total_steps < 60000:
            # 初始化游戏
            # routeId = random.randrange(0, 49, 1)
            routeId = 21
            self.routeline = self.allRoute[routeId]
            # print(self.routeline)
            env = Env(self.routeline)
            gameNum += 1
            # state = env.getState()  # 以state[0]、state[1]方式访问
            today = env.getToday()

            terminal = False
            order_accepted = False
            isExploration = False
            create_date = 1
            end_date = 0
            stay_num = 0
            # 一局游戏
            # print("GAME#:",gameNum)
            baseline = 0
            tao_prob = []
            tao_reward = []
            wait_day = []#记录一局游戏等待哪些天

            while today < self.routeline[-1] and terminal == False:
                # 有新订单产生 (当订单数已满10个时，此处不会收到新订单)
                if order_accepted == False:
                    self.orderSelect(self.routeline,60)
                    # print(self.order)
                    env.setOrder(self.order)
                    order_accepted = True
                # 遍历self.orders(即state[0])字典，对每一个订单操作
                state = env.getState()

                # 当前状态
                state_tf = np.mat(state)
                # print(state_tf,len(state_tf))
                # 由神经网络选择行动
                if random.random()<epsilon and isExploration == False:
                    isExploration = True
                    # end_date = random.randrange(env.getTodayIndex(),87,1)
                    end_date = 60

                if isExploration:
                    if env.getTodayIndex() == end_date:
                        action_model = 1
                        if ex_steps>0:
                            ex_steps -= 1
                    else:
                        action_model = 0
                else:
                    #action from learning
                    action_model = brain.choose_action(state_tf)
                    # print(action_model)

                wait_day.append(env.getTodayIndex())
                # 订单字典 历史曲线 reward

                reward = env.getReward(action_model)
                tao_reward.append(reward)
                # 订单完成或者到最后一天
                terminal = env.isTerminal(action_model)
                state_ = env.getNextState(action_model)
                if len(state_) == 1:
                    state_ = copy.deepcopy(state)
                brain.store_transition(state, action_model, reward, state_)
                # profitAdvanced_list.append(td_error[0][0])

                if brain.pointer > brain.MEMORY_CAPACITY :
                    # print(b_s_)
                    brain.learn()

                total_steps += 1
                if terminal:
                    # wait_list.append(wait_day[-1])
                    # loss = brain.learn()
                    # Loss_list.append(loss)
                    break

                # step 过一天加一
                env.nextStep()


            # 一局的总收益
            epsilon = self.epsilon*(ex_steps/500)
            print("epsilon:",epsilon)
            print("TD_Error:",baseline)
            profit = env.getTotalReward()
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
            if len(profit_list) >= 500:
                profit_list.clear()
                wait_list.clear()
            # last_remainder = total_steps % 1000

    def orderGenerator(price_mean, price_cov, time_mean, time_cov, size=1):
        price_axis = np.random.multivariate_normal(mean=price_mean, cov=price_cov, size=size)
        time_axis = np.random.multivariate_normal(mean=time_mean, cov=time_cov, size=size)
        # 需要再组合下两个，形成一个list
        return price_axis, time_axis

    def orderSelect(self,routeline,day_num):
        while True:
            # orderId = random.randrange(0, 59, 1)
            orderId = 0
            self.order = copy.deepcopy(self.allOrder[orderId])
            self.order[1] = '2019-06-20'
            orderDateStr = self.order[1]
            routeDateStr = routeline[-2]
            order_date = datetime.strptime(orderDateStr, "%Y-%m-%d")
            create_date = datetime.strptime(routeDateStr, "%Y-%m-%d")

            # print(orderDateStr,routeDateStr)
            # if orderDateStr < routeDateStr or order_date > create_date+ timedelta(days=40):
            if order_date > create_date + timedelta(days=day_num) or order_date < create_date + timedelta(days=25):
                continue
            else:
                # self.order = copy.deepcopy(self.allOrder[orderId])
                # self.order[1] = '2019-06-27'
                # self.order[1] = '2019-07-09'
                return
#读取订单数据
    def readOrder(self,filename):
        with open(filename, 'r') as f:
            content = f.readlines()
            result = []
            for c in content:
                list = []
                c = c.strip("[]\n\t")
                c = c.split(",")
                list.append(float(c[0]))
                c[1] = c[1].replace("'", "")
                list.append(c[1].replace(" ",""))
                result.append(list)
            # print(result[-1])
            return result
#读取航班数据
    def readRoute(self,filename):
        with open(filename, 'r') as f:
            content = f.readlines()
            result = []
            for c in content:
                list = []
                c = c.strip("[]\n")
                c = c.split(",")
                for i in range(89):
                    if i < 87:
                        list.append(float(c[i]))
                    else:
                        c[i] = c[i].replace("'", "")
                        list.append(c[i].replace(" ",""))
                result.append(list)
            # print(result)
            return result
#存储训练过程
    def writeHistory(self, filename, epsilon, baseline, total_steps, profit_list, profit, tao_prob, tao_reward, wait_day, gameNum):
        f = open(filename,'a')
        f.write("epsilon:"+ str(epsilon) + "\n")
        f.write("Baseline:"+ str(baseline)+"\n")
        f.write("total_steps:"+ str(total_steps)+"\n")
        f.write("profit_list"+ str(profit_list)+"\n")
        f.write("profit:"+ str(profit)+"profitAvg:"+ str(np.mean(profit_list))+"\n")
        f.write("action-prob:"+ str(tao_prob)+"\n")
        f.write("Reward:"+ str(tao_reward)+"\n")
        f.write("wait_day:"+ str(wait_day)+"\n")
        f.write("########################"+str(gameNum)+"###########################\n")
        f.flush()

    def testLSTM(self):
        brain = PolicyGradient(
            n_actions=2,
            n_features=87,
            learning_rate=0.02,
            reward_decay=0.85,
        )
        while True:
            # N = random.randrange(1,87,1)
            N=1
            list = self.random_int_list(1,3000,N)
            while len(list)<87:
                list.append(0)
            # print(np.mat(list)/2000)
            action_model = brain.choose_action(np.mat(list), N)
            print("length:",N,"action:",action_model)


    def random_int_list(self,start, stop, length):
        start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
        length = int(abs(length)) if length else 0
        random_list = []
        for i in range(length):
            random_list.append(random.randint(start, stop))
        return random_list

    def testRandomPolicy(self,day_num):
        game_num = 0
        profit_list = []
        while game_num < 5000:
            routeId = random.randrange(0, 49, 1)
            self.routeline = self.allRoute[routeId]
            self.orderSelect(self.routeline,60)
            buy_day = 1
            profit_list.append(self.order[0] - self.routeline[day_num-1])
            game_num += 1
        print("####################################")
        print("day_num",day_num)
        # print("Profit_List",profit_list)
        print("AVG_Profit",np.mean(profit_list))

if __name__ == '__main__':
    P = TikcetPlay()
    # for i in range(30,88):
    #     P.testRandomPolicy(i)
    P.transcate_DDPG()
    # P.testLSTM()
