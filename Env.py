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
        # 不限制最小值会导致没有起飞更早的航班
        # self.routeId = np.random.randint(low=self.his_t, high=len(self.data))
        self.routeId =21
        self.order_distribution = OrderGenerator(self.data[self.routeId], self.mode)
        self.totalReward = 0

        for i in range(self.his_t):
            for j in range(i + 1):
                if self.today + j < 87:
                    self.his_price[i, self.today + j] = self.data[self.routeId - i][self.today + j]
        self.his_order.append(self.order_distribution[self.today])

        state = {}
        state['buy_ticket_value'] = self.buy_ticket_value
        state['his_price'] = self.his_price
        state['his_order'] = self.his_order
        state['his_accept'] = self.his_accept
        state['orders'] = self.orders

        return state, self.done

    def step(self, act):
        """
        :param act: 0 hold, 1 accept order, 2 buy ticket for all orders
        :return: obs, reward , done , info
        """

        today_price = self.data[self.routeId][self.today]
        reward = 0
        info = {}

        order_accept = 0
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
        else:
            # 如果done了，return的state不使用，因此不更新也没事
            self.today += 1
            self.buy_ticket_value = 0
            self.his_order.append(self.order_distribution[self.today])
            today_price = self.data[self.routeId][self.today]
            for order in self.orders:
                self.buy_ticket_value += order - today_price
            for i in range(self.his_t):
                if self.today + i < 87:
                    self.his_price[i, self.today + i] = self.data[self.routeId - i][self.today + i]

        state = {}
        state['buy_ticket_value'] = self.buy_ticket_value
        state['his_price'] = self.his_price
        state['his_order'] = self.his_order
        state['his_accept'] = self.his_accept
        state['orders'] = self.orders

        return state, reward, self.done, info

    def separateStep(self, accpet_act, buy_act):
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

        if self.today >= 86:
            if type(buy_act).__name__ == 'list' or type(buy_act).__name__ == 'ndarray':
                buy_act = np.ones_like(buy_act)
            else:
                buy_act = 1
        reward_buy = self.getBuyReward(buy_act, self.orders)
        if len(self.orders) == len(buy_act):
            orders_after_action = []
            for i in range(len(buy_act)):
                if buy_act[i] == 0:
                    orders_after_action.append(self.orders[i])
            self.orders = orders_after_action
            profit = reward_buy
            self.profit += profit
        elif buy_act == 1:
            self.orders = []
        elif buy_act ==0:
            pass
        else:
            raise ValueError("len buy_act %d , len orders %d， not match" % (len(buy_act), len(self.orders)))

        if accpet_act == 1:
            if self.order_distribution[self.today] != -1 and self.order_left > 0:
                self.orders.append(self.order_distribution[self.today])
                order_accept = 1
                self.order_left -= 1
                reward_accept = self.getAcceptReward(accpet_act)
                # print("Accept:",self.today+1)a
            else:
                reward_accept = -1

        self.his_accept.append(order_accept)

        if self.today >= 86 or (len(self.orders) == 0 and self.order_left == 0):
            self.done = True
        else :
            self.today += 1
            self.buy_ticket_value = 0
            self.his_order.append(self.order_distribution[self.today])
            today_price = self.data[self.routeId][self.today]
            for order in self.orders:
                self.buy_ticket_value += order - today_price
            for i in range(self.his_t):
                if self.today + i < 87:
                    self.his_price[i, self.today + i] = self.data[self.routeId - i][self.today + i]

        state = {}
        state['buy_ticket_value'] = self.buy_ticket_value
        state['his_price'] = self.his_price
        state['his_order'] = self.his_order
        state['his_accept'] = self.his_accept
        state['orders'] = self.orders

        reward = {}
        reward['reward_accept'] = reward_accept
        reward['reward_buy'] = reward_buy

        return state, reward, self.done, info

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
        if type(act).__name__ == 'list' or type(act).__name__ == 'ndarray':
            for i in range(len(act)):
                # print("LOOK:", orders[i], today_price)
                if act[i] != 0:
                    if len(orders) >0:
                        reward += orders[i] - today_price
                    else:
                        reward -= 1
                else:
                    reward += 0
        elif act != 0 and len(orders) >= 1:
            for order in orders:
                reward += order - today_price
        elif act != 0 and len(orders) == 0:
            reward -= 1
        self.totalReward += reward
        return reward

    def getTotalReward(self):
        return self.totalReward

    def getTodayIndex(self):
        if self.order_num == self.order_left:
            return -1
        else:
            return self.today

    def getNextPrice(self):
        next_price = np.zeros((self.his_t, 87))
        if self.today >= 86:
            next_price = self.his_price
        else:
            for i in range(self.his_t):
                if self.today + i < 87:
                    next_price[i, self.today + i] = self.data[self.routeId - i][self.today + i]
        return next_price



if __name__ == "__main__":
    route_list = readRoute("./wang/data/route")
    env = Env(route_list,history_take_off=2)
    for i in range(86):
        action = np.random.randint(3)
        obs, reward, done, info = env.step(action)
        env.render()
        print(obs["his_price"])