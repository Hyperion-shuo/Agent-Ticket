# coding=utf-8
from connectMysql import OrderMysql, RouteMysql, RecordMysql, MoneyMysql, GameRecordMysql, MeanMysql
from entity.baseClass import Order
import random
import numpy as np
from datetime import*


class GameState:
    def __init__(self, routeLine, routeList):
        # 连接数据库
        '''
        self.om = OrderMysql()
        self.routem = RouteMysql()
        self.recordm = RecordMysql()
        self.mm = MoneyMysql()
        self.gm = GameRecordMysql()
        self.msm = MeanMysql()
        '''
        # 每局游戏的初始数据，不用更新
        # List 每个元素是长为87的 route 对象的list
        self.route_list = routeList
        # 当前航线名称 字符串
        self.routeLine = routeLine
        # 游戏局数，用于数据库中记录作为主键之一，重复使用要清空game_record表
        # self.gameNum = gameNum
        ####################随机取routeId#########################
        # self.routeId = random.randrange(49) # list 从 0 到 127
        self.routeId = 22
        self.departDate = self.route_list[self.routeId][0].departDate
        
        # 订单相关数据
        # 每天的随机订单列表
        self.random_order_list = []
        # 已接订单列表，存放order对象，已完成的也在，is_finsih置0
        self.order_list = []
        # 订单字典，key是order对象的id（1，10），value是订单价格
        self.orders = {}
        # 当前日期的可接订单字典, 是否存在与价格，每天更新
        self.newOrder = {"existence": 0, "price": 0}
        # 当前订单数
        self.orderNum = 0

        # route对象的信息，每天更新
        self.day = 1  # 注意用作下标查询list 时要减 1
        # 当前日期、出发日期、第一天价格加入self.historicalPrice列表
        self.date = self.route_list[self.routeId][0].date
        self.todayPrice = self.route_list[self.routeId][0].price
        # 价格曲线列表
        self.historicalPrice = []
        self.historicalPrice.append(self.todayPrice)

        # 计算得到的所有航线上的价格均值与方差，用于标准化
        # 现在未使用
        self.mean, self.std = 1176.12, 472.6

        # 随机生成的order，每天都有, 与数据库中的生成方法一致
        # i从0开始，等于day - 1
        for i in range(87):
            # 计算当前日期前航班的日平均价格 avg_price
            sum_price = 0
            j = 0
            while j <= i:
                sum_price += self.route_list[self.routeId][j].price
                j += 1
            avg_price = sum_price / (i + 1)  # i，j都从0开始

            # 减的部分视为难度调整
            # avg_price -= random.randint(0, 1000)

            # 如果平均价格高于当日价格则订单等于平均价格
            # 如果更低则用当日价格减一些替代
            if avg_price < self.route_list[self.routeId][i].price:
                price = avg_price
            else:
                # price = self.route_list[self.routeId][i].price - random.randint(0, 1000)
                price = self.route_list[self.routeId][i].price
            self.random_order_list.append(price)

        # 第一天的随机订单, 一个数字 ，价格
        self.newOrder["price"] = self.random_order_list[self.day - 1]
        self.newOrder["existence"] = 1

    # 返回接单行动后的下一个状态
    def acceptOrders(self, input_actions):

        # input_actions（0,0）取值为0,1
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!动作输入错误')

        # input_actions[0] == 1: 不接单
        # input_actions[1] == 1: 接单
        if input_actions[1] == 1 and self.orderNum < 10:

            # 计算最大利润，和平均利润（在当前天后随机选一天买票的利润）
            temp_day = self.day
            sum_price_after = 0
            lowest_price_after = self.route_list[self.routeId][self.day - 1].price
            while temp_day <= 87:
                today_price = self.route_list[self.routeId][temp_day - 1].price
                sum_price_after += today_price  # avg_price 包含今天的价格因为可以选择当天买票
                if today_price < lowest_price_after:
                    lowest_price_after = today_price
                temp_day += 1
            # sum_price 有today_price 因此分母要加1
            avg_price_after = sum_price_after / (87 - self.day + 1)
            # 读取订单价格、未来最低价，计算最大收益
            price = self.newOrder["price"]
            avgProfit = price - avg_price_after
            maxProfit = price - lowest_price_after

            # 新订单 isFinished 和 profit 默认置 0, orderNum 就是 orderId 从 1 到 10
            newOrder = Order(self.orderNum, self.departDate, self.routeLine, price, self.date, maxProfit, avgProfit)
            # 更新游戏中order相关的数据
            self.order_list.append(newOrder)
            self.orderNum += 1

            # 从tb_order中读取未完成的订单价格列表
            self.orders = {}
            for order in self.order_list:
                if order.isFinished == 0:
                    self.orders[str(order.orderId)] = order.price  # 未测试

            # 看order数量，没加天数
            # print("order_num:%d"%(len(self.order_list)))

        return self.orders, self.historicalPrice

    # 完成单个订单
    def finishOrders(self, orderId, input_actions, day):
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!动作输入错误')
        '''
        # 最后一天自动买票的软惩罚
        if self.date == self.departDate - timedelta(days=1) and input_actions[0] == 1:
            profit = self.om.finishOeder(orderId, self.todayPrice)
            reward = profit - 100
            self.orders = self.om.getOrderPriceRemain()
        '''
        reward = 0
        if input_actions[1] == 1:
            # 完成一个订单. orderId 0 到 9  ，字典里的key “orderId” 变为数组下标
            profit = self.order_list[int(orderId)].price - self.route_list[self.routeId][self.day - 1].price
            self.order_list[int(orderId)].profit = profit
            self.order_list[int(orderId) ].isFinished = 1

            # 这个reward 打算找出后面有多少个比这个价格还低的
            # reward = self.routem.lowerPriceAfter(self.date, self.routeId, self.todayPrice)

            # 这是最直接的 reward
            reward = profit

            # 这是股价预测中的常用reward
            # reward = np.log10(self.order_list[int(orderId) - 1].price / self.route_list[self.routeId][self.day - 1].price)

            # 从tb_order中读取未完成的订单价格列表
            self.orders = {}  # 先重置再写进去
            for order in self.order_list:
                if order.isFinished == 0:
                    self.orders[str(order.orderId)] = order.price

            print("buy ticket at %s" % day)
        else:
            # 给等待分级惩罚，希望尽早买票
            reward = 0

        return self.orders, self.historicalPrice, reward

    # 进入新一天
    def nextDay(self):
        # 更新日期、当天价格、价格曲线
        self.date = self.date + timedelta(days=1)
        self.day += 1
        self.todayPrice = self.route_list[self.routeId][self.day - 1].price
        self.historicalPrice.append(self.todayPrice)
        # 读取新订单
        # self.newOrder = {"existence": 0, "price": 0}
        if self.orderNum < 10:
            self.newOrder["price"] = self.random_order_list[self.day - 1]
            self.newOrder["existence"] = 1

    def getTomorrowPrice(self):
        tomorrow = self.day + 1
        tomorrowPrice = self.route_list[self.routeId][tomorrow - 1].price

        return tomorrowPrice

    # 当前这局游戏的已完成订单的
    # 实际总收益, 随机买票收益, 最大收益
    def getProfit(self):
        totalProfit, avgProfit, maxProfit = 0, 0, 0
        for order in self.order_list:
            if order.isFinished == 1:
                totalProfit += order.profit
            avgProfit += order.avgProfit
            maxProfit += order.maxProfit
        return totalProfit, avgProfit, maxProfit

if __name__ == "__main__":
    GS = GameState(1, 1, )  # 还要输入所有route信息
    print(GS.route_list.__len__())