import numpy as np
from datetime import datetime, date
from datetime import timedelta
import copy
from OrderGenerate import OrderGenerator, readRoute
import matplotlib.pyplot as plt

class Env():

    def __init__(self, data, mode=1, history_take_off=2,order_num = 10):
        self.data = data
        self.his_t = history_take_off
        self.mode = mode
        self.his_order = []
        self.his_accept = []
        self.order_num = order_num
        self.totalReward = 0.5
        self.reset()

    def reset(self):
        self.today = 0
        self.done = False
        self.profit = 0
        self.orders = []
        self.order_left = self.order_num
        self.his_order = []
        self.his_accept = []
        self.buy_ticket_value = 0
        self.his_price = np.zeros((self.his_t, 87))
        # self.routeId = np.random.randint(len(self.data))
        self.routeId =21
        self.order_distribution = OrderGenerator(self.data[self.routeId], self.mode)
        self.totalReward = 0

        for i in range(self.his_t):
            for j in range(i + 1):
                if self.today + j < 87:
                    self.his_price[i, self.today + j] = self.data[self.routeId - i][self.today + j]


        return (self.buy_ticket_value, self.his_price, self.his_order, self.his_accept)

    def step(self, act):
        """
        :param act: 0 hold, 1 accept order, 2 buy ticket for all orders
        :return: obs, reward , done , info
        """

        today_price = self.data[self.routeId][self.today]
        reward = 0
        info = {}

        order_accept = 0
        self.his_order.append(self.order_distribution[self.today])
        if act == 1:
            if self.order_distribution[self.today] != -1 and self.order_left > 0:
                self.orders.append(self.order_distribution[self.today])
                order_accept = 1
                self.order_left -= 1
                reward = self.getAcceptReward(act)
        elif act == 2:
            profit = 0
            reward = self.getBuyReward(act, self.orders)
            profit = reward
            self.profit += profit
            self.orders = []

        self.his_accept.append(order_accept)


        if self.today >= 86 or (len(self.orders) == 0 and self.order_left == 0):
            self.done = True
        self.today += 1
        self.buy_ticket_value = 0
        for order in self.orders:
            self.buy_ticket_value += order - today_price
        for i in range(self.his_t):
            if self.today + i < 87:
                self.his_price[i, self.today + i] = self.data[self.routeId - i][self.today + i]

        return (self.buy_ticket_value, self.his_price, self.his_order, self.his_accept), reward, self.done, info

    def SeparateStep(self, accpet_act, buy_act):
        '''
        :param accpet_act: 0 hold, 1 accept order
        :param buy_act: a list for each orders action, 0 hold, 1 buy
        :return: obs, reward , done , info
        '''

        today_price = self.data[self.routeId][self.today]
        reward_accept = 0
        reward_buy = 0
        info = {}

        order_accept = 0
        self.his_order.append(self.order_distribution[self.today])

        if accpet_act == 1:
            if self.order_distribution[self.today] != -1 and self.order_left > 0:
                self.orders.append(self.order_distribution[self.today])
                order_accept = 1
                self.order_left -= 1
                reward_accept = self.getAcceptReward(accpet_act)


        if len(self.orders)==len(buy_act):
            reward_buy = self.getBuyReward(buy_act, self.orders)
            for i in range(len(buy_act)):
                if buy_act[i] == 1 :

                    self.orders.pop(i)
            profit = reward_buy
            self.profit += profit

        self.his_accept.append(order_accept)


        if self.today >= 86 or (len(self.orders) == 0 and self.order_left == 0):
            self.done = True
        self.today += 1
        self.buy_ticket_value = 0
        for order in self.orders:
            self.buy_ticket_value += order - today_price
        for i in range(self.his_t):
            if self.today + i < 87:
                self.his_price[i, self.today + i] = self.data[self.routeId - i][self.today + i]

        return (self.buy_ticket_value, self.his_price, self.his_order, self.his_accept), (reward_accept,reward_buy), self.done, info

    def render(self):
        print(f'Day: {self.today}')
        print(f'Orders: {self.orders}')
        print(f'Order left: {self.order_left}')
        print(f'Profit: {self.profit}')

    def getAcceptReward(self, act):
        '''
        :param act: 0 reject, 1 accept
        :return: reward
        '''
        reward = 0
        return reward

    def getBuyReward(self, act, orders):
        '''
        :param act: 0 hold, 1 buy
        :return: reward
        '''
        reward = 0
        today_price = self.data[self.routeId][self.today]
        if type(act).__name__ == 'list':
            for i in range(len(orders)):
                # print("LOOK:", orders[i], today_price)
                if act[i] != 0:
                    reward += orders[i] - today_price
                else:
                    reward += 0
        elif act != 0 and len(orders) >= 1:
            for order in orders:
                reward += order - today_price
        self.totalReward += reward
        return  reward

    def getTotalReward(self):
        return self.totalReward

    def getTodayIndex(self):
        if self.order_num == self.order_left:
            return -1
        else:
            return self.today



if __name__ == "__main__":
    route_list = readRoute("./wang/data/route")
    env = Env(route_list,history_take_off=1)
    for i in range(86):
        action = np.random.randint(3)
        obs, reward, done, info = env.step(action)
        env.render()
        print(obs[1])