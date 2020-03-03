# coding=utf-8
from GameState import GameState
from algorithm.BrainDQN import BrainDQN
from connectMysql import OrderMysql, RouteMysql, RecordMysql, MoneyMysql, GameRecordMysql, MeanMysql
import numpy as np
import random
from datetime import *
import matplotlib.pyplot as plt

MAX_STEP = 40010
REPLAY_MEMORY = 10005

# 目前aceeptorder里 finish_order 里没有往game表中存储

def Avg(profit_list):
    num = 0
    sum = 0
    if len(profit_list) > 0:
        for profit in profit_list:
            sum += profit
            num += 1
        avg = sum / num
        return avg

# 把游戏环境返回的状态转换为输入神经网络的数组
# 更改state 要修改优先存储处的sample 函数
def to_state_tf(state, key, day):
    state_tf = []  # 列表
    # state_tf.append(state[0][key])  # 填补少的第一点， 原为订单价格， 保持长度不变
    # List extend numpy数组会自动一个个插入,而不是把numpy数组当做一个元素
    state_tf.extend(np.array(state[1]) - state[0][key])
    while len(state_tf) < 88:
        state_tf.append(0)
    # 变为矩阵，适应网络输入
    state_tf = np.array(state_tf)
    # state_tf[:day] = (state_tf[:day] - mean) / std # 标准化 eg. day1 state[0]被标准化
    state_tf = np.reshape(state_tf, [88, 1])  # 88 变 88 * 1

    d = np.zeros(88).reshape([88, 1])  # 现在是第几天的one hot 向量
    d[day - 1, 0] = 1  # 第一天第0个位置为1

    # 88 * 2 88行，第一列为 state_tf 第二列为 d
    state_tf = np.concatenate((state_tf, d), axis=1)
    # 加一维代表通道数，卷积用 88 * 2 * 1 实际在输入网络是最前面加了一维，即代表样本数的那一维
    state_tf = state_tf[:, :, np.newaxis]

    return state_tf


# 同上，先找到下一天的价格，再把游戏环境的状态转为输入神经网络的数组
def to_state_next(state, key, transacte, date, day):
    state_next = []
    # state_next.append(state[0][key])  # 订单价格
    state_next.extend(np.array(state[1]) - state[0][key])  # 历史价格曲线
    # 此处再看，对最后一天的处理
    if date < transacte.departDate - timedelta(days=1):
        tomorrowPrice = transacte.getTomorrowPrice()
        state_next.append(tomorrowPrice - state[0][key])  # 这里注意
        # state_next.append(tomorrowPrice)

    while len(state_next) < 88:
        state_next.append(0)
    state_next = np.array(state_next)
    '''
    if day < 87:
        state_next[:day + 1] = (state_next[:day + 1] - mean) / std # 第一天state[0] 处被标准化
    else:
        state_next[:day] = (state_next[:day] - mean) / std  # 87天没有后一天 不能对补空的0做归一化
    '''
    state_next = np.reshape(state_next, [88, 1])

    d = np.zeros(88).reshape([88, 1])
    d[day, 0] = 1  # 第一天的nextday 数组第二个位置下标1为0

    state_next = np.concatenate((state_next, d), axis=1)
    state_next = state_next[:, :, np.newaxis]

    return state_next

# 多个episode 订单平均实际收益和平均随机收益
def plot_ret(profit_list, profit_random_list, total_steps, avg_profit, epsilon):
    plt.plot(profit_list, "b-", label="agent")
    plt.plot(profit_random_list, "r-", label="random")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.title("avgreward:" + str(avg_profit) + " " + "epsilon:" + str(epsilon))
    plt.savefig('./picture/' + "Reward" + "_steps_" + str(total_steps / 10000) + '_.png')
    plt.cla()

# 多个episode的平均买票天数
def plot_day(avg_day_list, total_steps, avg_day, epsilon):
    plt.plot(avg_day_list,label="avgday")
    plt.title("AvgDay" + str(avg_day) + " epsilon" + str(epsilon))
    plt.xlabel("episode")
    plt.ylabel("day")
    plt.legend()
    plt.savefig(
        './picture/' + "Avgday" + "_Steps_" + str(total_steps / 10000) + '_.png')
    plt.cla()


def playTicketTransacte():
    # 从数据库里取出航线信息
    routem = RouteMysql()
    route_list = []  # 每个元素是个长为87的route对象数组
    for i in range(1, 129):  # 0 到 127 共 128 条
        one_route_list = routem.getOneRoute(i)
        route_list.append(one_route_list)
    routem.__exit__()

#############DebugPrint############
    '''
    # 查看所有route数据
    print(route_list.__len__())
    print(route_list[0].__len__())
    print(dir(route_list[1][0]))
    for i in range(128):
        for j in range(87):
            print(str(route_list[i][j].routeId) + " " + str(route_list[i][j].departDate) + " "\
                  + str(route_list[i][j].date) + " " + str(route_list[i][j].price))
    '''
###################################

    # 初始化RL Brain
    actions = 2  # 行动个数
    # f = open(str(initial_epsilon) + ".txt", 'w', encoding="utf8")
    brain = BrainDQN(actions, prioritized=True)

    # 记录游戏数据
    total_steps = 0  # 记录步数，一天一个订单的判断是一步
    game_num = 0
    profit_list, avg_day_list, profit_random_list = [], [], []

    while total_steps < MAX_STEP:
        # 初始化游戏
        game_num += 1
        transacte = GameState("BJS-DLU", route_list)

        # 初始化游戏内数据
        # state[0]订单key是id，value是值， state[1]历史价格序列
        state = transacte.orders, transacte.historicalPrice
        date, day = transacte.date, transacte.day # day 1 到 87
        day_list, order_finish_day = [], []

        # 整个episode 用来探索
        # 每天随机买票
        is_explore = 1
        if np.random.uniform() < brain.epsilon:
            is_explore = 1
            while date < transacte.departDate:
                if transacte.newOrder["existence"] == 1:

                    action_acceptOrder = np.zeros(2)

                    # action_index = random.randrange(5)  # 0-4范围, 1/5概率选择


                    # 只接第一天的订单
                    if day == 1:
                        action_index = 0
                    else :
                        action_index = 1


                    #  必然接单
                    # action_index = 0

                    if action_index > 0:
                        action_index = 0
                        action_acceptOrder[action_index] = 1
                        state = transacte.acceptOrders(action_acceptOrder)
                    else:
                        action_index = 1
                        action_acceptOrder[action_index] = 1
                        state = transacte.acceptOrders(action_acceptOrder)
                        # 找到新加的订单，设好随机买票的天数
                        newest_order = 0
                        for key in state[0]:
                            if int(key) > newest_order:  # dict 中的key变为int作list的下标用
                                newest_order = int(key)
                        delta = transacte.order_list[newest_order - 1].departDate - transacte.order_list[newest_order - 1] \
                            .createDate
                        interval = delta.days  # 转换类型 timedelta 变为 int
                        # 随机买票， 不能取到Interval，那一天价格数据库中没有
                        order_finish_day.append(day + np.random.randint(0, interval))
                        # print("newest_order" + " " + str(newest_order))
                        # print("len" + " " + str(order_finish_day.__len__()))
                        # print(order_finish_day)
                # 处理订单模型，遍历self.orders(即state[0])字典，对每一个订单操作
                state_ = state
                for key in state[0]:  # key 是orderId

                    state_tf = to_state_tf(state, key, day)

                    if order_finish_day[int(key) - 1] == day:  # 查看是否到达随机买票日期
                        action_finishOrder = [0, 1]
                    else:
                        action_finishOrder = [1, 0]
                    if action_finishOrder[1] == 1:
                        day_list.append(day)

                    state_0, state_1, reward = transacte.finishOrders(key, action_finishOrder, day)
                    state_ = state_0, state_1

                    state_next = to_state_next(state, key, transacte, date, day)

                    # 是否终止
                    terminal = False

                    if action_finishOrder[1] == 1 or date == transacte.departDate - timedelta(days=1):
                        terminal = True

                    brain.store_transition(state_tf, action_finishOrder, reward, state_next, terminal)

                    total_steps += 1

                    # 500天输出一次
                    # if total_steps >= REPLAY_MEMORY:
                    if total_steps >= REPLAY_MEMORY:
                        # 控制memory里每个元素被采样次数的期望
                        # if total_steps % 10 == 0:
                        brain.trainQNetwork()

                        if total_steps % 10000 == 0:
                            # 展示总收益曲线
                            plot_ret(profit_list, profit_random_list, total_steps, Avg(profit_list), brain.epsilon)
                            plot_day(avg_day_list, total_steps, Avg(avg_day_list), brain.epsilon)
                            '''
                            profit_list.clear()
                            avg_day_list.clear()
                            profit_random_list.clear()
                            '''

                #
                state = state_
                # 进入新一天，date在一开始从transacte中得到
                date = date + timedelta(days=1)
                day += 1
                if date < transacte.departDate:
                    transacte.nextDay()
        else:
            is_explore = 0
            # < 而非 <= 因为数据库中没有起飞那一天的数据
            while date < transacte.departDate:
                # 接单模型， 有新订单产生 (当订单数已满10个时，此处不会收到新订单)
                if transacte.newOrder["existence"] == 1:
                    # 随机选择行动产生
                    action_acceptOrder = np.zeros(2)

                    '''
                    # [0,1]--0.2或者[1,0]--0.8
                    action_index = random.randrange(5)  # 0-4范围, 1/5概率选择
                    if action_index > 0:
                        action_index = 0
                    else:
                        action_index = 1
                    '''



                    # 只接第一天订单
                    if day == 1:
                        action_index = 1
                    else :
                        action_index = 0



                    # 必然接单
                    # action_index = 1

                    action_acceptOrder[action_index] = 1
                    state = transacte.acceptOrders(action_acceptOrder)

                # 处理订单模型，遍历self.orders(即state[0])字典，对每一个订单操作
                state_ = state
                # 无order则不执行
                for key in state[0]:
                    # 把列表变成可传入神经网络的矩阵
                    state_tf = to_state_tf(state, key, day)

                    # 由神经网络选择行动
                    # print("order price : " + str(transacte.orders[key]))
                    action_finishOrder = brain.getAction(state_tf, day)
                    if action_finishOrder[1] == 1:
                        day_list.append(day)
                    # 订单字典 历史曲线 reward
                    state_0, state_1, reward = transacte.finishOrders(key, action_finishOrder, day)
                    state_ = state_0, state_1

                    # 该订单的下一个状态，存入四元组用
                    state_next = to_state_next(state, key, transacte, date, day)
                    # state_next = state_next / denominator

                    # 是否终止
                    terminal = False
                    # 订单完成或者到最后一天
                    if action_finishOrder[1] == 1 or date == transacte.departDate - timedelta(days=1):
                        terminal = True
                    # 保存记录到记忆库
                    brain.store_transition(state_tf, action_finishOrder, reward, state_next, terminal)

                    # step 对每个订单决策一次加1
                    total_steps += 1

                    if total_steps >= REPLAY_MEMORY:
                        # 控制memory里每个元素被采样次数的期望
                        # if total_steps % 10 == 0:
                        brain.trainQNetwork()

                        if total_steps % 10000 == 0:
                            # 展示总收益曲线
                            plot_ret(profit_list, profit_random_list, total_steps, Avg(profit_list), brain.epsilon)
                            plot_day(avg_day_list, total_steps, Avg(avg_day_list), brain.epsilon)
                            '''
                            profit_list.clear()
                            avg_day_list.clear()
                            profit_random_list.clear()
                            '''
                #
                state = state_
                # 进入新一天，date在一开始从transacte中得到
                date = date + timedelta(days=1)
                day += 1
                if date < transacte.departDate:
                    transacte.nextDay()

        # 记录一个episode 的信息
        profit, profit_random, profitmax = transacte.getProfit()
        profit_list.append(int(profit))
        profit_random_list.append(int(profit_random))
        avg_day_list.append(Avg(day_list))
        if is_explore:
            print("Explore game_num:", game_num, "profit:", profit, "profitAvg:", Avg(profit_list))
        else:
            print("Agent game_num:", game_num, "profit:", profit, "profitAvg:", Avg(profit_list))




def main():
    playTicketTransacte()


if __name__ == '__main__':
    main()
