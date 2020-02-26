import numpy as np
from datetime import datetime, date
from datetime import timedelta
import copy
class Env():
    def __init__(self, routeline):
        self.routeline = routeline[:87]
        self.start_date = routeline[-2]
        self.state = np.zeros_like(self.routeline)
        self.state[0] = self.routeline[0]
        self.order = []
        self.departDateProperty = self.getDepartDateProperty(routeline[-1])
        self.waiteDate = 0
        self.today = 1
        self.totalReward = 0
        # self.rewardList = []

    def setOrder(self,order):
        self.order = order
        startDate = datetime.strptime(self.start_date, "%Y-%m-%d")
        orderDate = datetime.strptime(self.order[1], "%Y-%m-%d")
        while startDate < orderDate:
            self.nextStep()
            startDate += timedelta(days=1)


    def getTotalReward(self):
        return self.totalReward

    def getTodayIndex(self):
        return self.today

    def getState(self):
        exist = np.zeros(self.today)
        for t in range(0, self.today):
            exist[t] = self.state[t]
        # exist -= np.mean(exist)
        # exist /= np.std(exist)
        exist -= 1877.368
        exist /= 256.61
        exist = exist.tolist()
        while len(exist) < 87:
            exist.append(0)
        # print("STATE:", self.state,"EXIST:",exist)
        return exist

    def getNextState(self,action):
        if self.today>=87 or action == 1:
            return [-1]
        exist = np.zeros(self.today+1)
        for t in range(0, self.today+1):
            exist[t] = self.routeline[t]
        # exist -= np.mean(exist)
        # exist /= np.std(exist)
        exist -= 1877.368
        exist /= 256.61
        exist = exist.tolist()
        while len(exist) < 87:
            exist.append(0)
        # print("STATE:", self.state,"EXIST:",exist)
        return exist

    def getToday(self):
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        today = start_date + timedelta(days=self.today-1)
        return today.strftime("%Y-%m-%d")

    def isTerminal(self,action):
        if self.today == 87 or action == 1:
            return True
        else:
            return False

    def nextStep(self,isOrderAccept=False):
        self.state[self.today] = self.routeline[self.today]
        self.today += 1
        if isOrderAccept:
            self.waiteDate += 1

    def getReward(self,action):
        if action == 0 and self.today < 87:
            # reward = self.state[self.today - 1] - self.order[0]
            # self.totalReward += reward
            reward = 0
            return reward
        else:
            #直接盈利作为reward
            reward = self.order[0] - self.state[self.today-1]
            self.totalReward += reward
            #需要跟后面真实收益进行比较的操作
            # today_price = self.state[self.today - 1]
            # waite_list = self.routeline[self.today-self.waiteDate-1:]
            # # print("起始点：",self.today)
            # waite_list.sort()
            # index = waite_list.index(today_price)
            # #归一化(回合内）之后的结果
            # waite_list = [self.order[0] - i for i in waite_list]
            # print("Wait:",waite_list)
            # # print("State:",self.state)
            # todayProfit = self.order[0] - today_price
            # maxProfit = waite_list[0]
            # minProfit = waite_list[-1]
            # # print("today:",todayProfit,",min:",minProfit,",max:",maxProfit)
            # if minProfit < 0:
            #     if todayProfit > 0:
            #         reward = todayProfit / maxProfit
            #     else:
            #         reward = -1.0 * todayProfit / minProfit
            # else:
            #     reward = todayProfit / (maxProfit - minProfit)

            #第几顺位作为reward
            # reward = 1 - (index/(len(waite_list)-1))
            # reward =  waite_list[0] / today_price
            # print("reward:",reward)
            # self.totalReward += todayProfit
            return reward

# 直接盈利作为reward
    def getProfit(self,action):
        if action == 0 and self.today < 87:
            reward = 0
            return reward
        else:
            today_price = self.state[self.today - 1]
            waite_list = self.routeline[self.today-self.waiteDate-1:]
            waite_list.sort()
            index = waite_list.index(today_price)
            self.totalReward += reward
            return reward
#上帝视角可赚钱的排序
    def getProfit(self,action):
        if action == 0 and self.today < 87:
            reward = 0
            return reward
        else:
            reward = self.order[0] - self.state[self.today-1]
            self.totalReward += reward
            return reward

    def getRewardShaping(self):
        if self.today == 1:
            reward = self.state[0] - self.order[0]
            self.totalReward += reward
            return reward
        else:
            reward = self.state[self.today-2] - self.state[self.today-1]
            self.totalReward += reward
            return reward

    def getDepartDateProperty(self, str):
        hol = {"2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-06-07", "2019-06-08", "2019-06-09",
               "2019-09-13"}
        work = {"2019-04-28", "2019-05-05"}
        date = datetime.strptime(str,"%Y-%m-%d")
        #周中+工作日：0
        #周中+休息日：1
        #周末+工作日：2
        #周末+休息日：3
        label = 0
        if date.weekday() < 5:
            label += 0
        else:
            label += 2
        if str in work:
            label += 0
        else:
            label += 1

        return label