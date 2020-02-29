import numpy as np
from datetime import datetime, date
from datetime import timedelta
import copy
from OrderGenerate import OrderGenerator, readRoute
import matplotlib.pyplot as plt

class Env():

    def __init__(self, data, mode=1, history_take_off=2):
        self.data = data
        self.his_t = history_take_off
        self.mode = mode
        self.reset()

    def reset(self):
        self.today = 0
        self.done = False
        self.profit = 0
        self.orders = []
        self.order_left = 10
        self.buy_ticket_value = 0
        self.history = np.zeros((self.his_t, 87))
        # self.routeId = np.random.randint(len(self.data))
        self.routeId =1
        self.order_distribution = OrderGenerator(self.data[self.routeId], self.mode)

        for i in range(self.his_t):
            for j in range(i + 1):
                if self.today + j < 87:
                    self.history[i, self.today + j] = self.data[self.routeId - i][self.today + j]


        return self.history

    def step(self, act):
        """
        :param act: 0 hold, 1 accept order, 2 buy ticket for all orders
        :return: obs, reward , done , info
        """

        today_price = self.data[self.routeId][self.today]
        reward = 0
        info = {}

        if act == 1:
            if self.order_distribution[self.today] != -1 and self.order_left > 0:
                self.orders.append(self.order_distribution[self.today])
                self.order_left -= 1
        elif act == 2:
            if len(self.orders) == 0:
                reward = 0
            else:
                profit = 0
                for order in self.orders:
                    profit += order - today_price
                reward += profit
                self.profit += profit
                self.orders = []

        if self.today >= 86 or (self.orders == 0 and self.order_left == 0):
            self.done = True
        self.today += 1
        self.buy_ticket_value = 0
        for order in self.orders:
            self.buy_ticket_value += order - today_price
        for i in range(self.his_t):
            if self.today + i < 87:
                self.history[i, self.today + i] = self.data[self.routeId - i][self.today + i]

        return (self.buy_ticket_value, self.history), reward, self.done, info

    def render(self):
        print(f'Day: {self.today}')
        print(f'Orders: {self.orders}')
        print(f'Order left: {self.order_left}')
        print(f'Profit: {self.profit}')


if __name__ == "__main__":
    route_list = readRoute("./wang/data/route")
    env = Env(route_list)
    for i in range(86):
        action = np.random.randint(3)
        obs, reward, done, info = env.step(action)
        env.render()
        print(obs[0])