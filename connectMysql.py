# coding=utf-8
import pymysql
import numpy as np
import matplotlib.pyplot as plt
from  datetime import*
from entity.baseClass import Money
from entity.baseClass import Route
from entity.baseClass import Order
import random


class ConnectMysql():
    def __init__(self, host='localhost', port=3306, db='game', user='root', passwd='123456789', charset='utf8'):
        # 建立连接
        self.conn = pymysql.connect(host=host, port=port, db=db, user=user, passwd=passwd, charset=charset)
        # 创建游标，操作设置为字典类型
        self.cur = self.conn.cursor(cursor = pymysql.cursors.DictCursor)

    def __enter__(self):
        # 返回游标
        return self.cur

    def __exit__(self):
        # 提交数据库并执行
        self.conn.commit()
        # 关闭游标
        self.cur.close()
        # 关闭数据库连接
        self.conn.close()



# 对tb_money表进行读、写操作， 继承自ConnectMysql类
class MoneyMysql(ConnectMysql):

    # 重置游戏
    def reset(self, date):
        sql = "update tb_money set money = 0, days = 1, today = '%s' where moneyId = 1" % (date)

        self.cur.execute(sql)
        self.conn.commit()

    # 有订单完成，更新表头(更改money)
    def update(self, money, price, todayPrice):
        moneyNow = money.getMoney() + (price - todayPrice)

        sql = "update tb_money set money = '%s' where moneyId = 1" \
              % (moneyNow)
        self.cur.execute(sql)
        self.conn.commit()

    # 进入下一天， 更新日期(更改days 和 today)
    def nextDay(self, money):
        days = money.getDays() + 1
        today = money.getToday()
        _datetime_today = datetime.strptime(today, '%Y-%m-%d') # 变成日期格式
        _tomorrow = _datetime_today + timedelta(days=1)
        tomorrow = _tomorrow.strftime('%Y-%m-%d')

        sql = "update tb_money set days = '%s',today = '%s' where moneyId = 1" \
                %(days, tomorrow)
        self.cur.execute(sql)
        self.conn.commit()

#对tb_route、tb_random_order进行读写
class RouteMysql(ConnectMysql):
    # 读取当前天route
    def getOneRoute(self, routeId):
        sql = "SELECT * from tb_route where routeId = '%d' order by date" % (routeId)

        self.cur.execute(sql)
        rets = self.cur.fetchall()

        one_route_list = []
        for ret in rets:
            routeret = Route()
            try:
                routeret.setRouteId(ret["routeId"])
                routeret.setDepartDate(ret["departDate"])
                routeret.setPrice(ret["price"])
                routeret.setDate(ret["date"])
                routeret.setRouteLine(ret["routeLine"])
                one_route_list.append(routeret)
            except:
                print("Error: unable to fetch data")

        return one_route_list


    def nowRoute(self, date, routeId):
        sql = "SELECT * from tb_route where date='%s' and routeId='%d'" \
              %(date,routeId)
        routeret = Route()
        try:
            self.cur.execute(sql)
            ret = self.cur.fetchone()
            routeret.setRouteId(ret["routeId"])
            routeret.setDepartDate(ret["departDate"])
            routeret.setPrice(ret["price"])
            routeret.setDate(ret["date"])
            routeret.setRouteLine(ret["routeLine"])
        except:
            print("Error: unable to fetch data")

        return routeret
    # 获取第一天
    def getFirstDate(self, routeId):
        sql = "SELECT * from tb_route where routeId='%d'" \
              % (routeId)
        try:
            self.cur.execute(sql)
            ret = self.cur.fetchone()
            firstDate = ret["date"]
            departDate = ret["departDate"]
            price = ret["price"]
            return firstDate, departDate, price
        except:
            print("Error: unable to fetch data")


    # 读取至今所有天route
    def allRoute(self, date, routeId):
        sql = "SELECT routeId,departDate,price,date,routeLine from tb_route where date<='%s' and routeId='%d'" \
              % (date, routeId)
        self.cur.execute(sql)
        ret = self.cur.fetchall()
        return ret

    '''
    # 读取至今所有天route
    def allRoute(self, routeId):
        sql = "SELECT routeId,departDate,price,date,routeLine from tb_route where routeId='%d' order by date" \
              % (routeId)
        self.cur.execute(sql)
        ret = self.cur.fetchall()
        return ret
    '''

    def allRouteLength(self, routeId):
        sql = "SELECT routeId,departDate,price,date,routeLine from tb_route where routeId='%d'" \
              % (routeId)
        self.cur.execute(sql)
        ret = self.cur.fetchall()
        return ret.__len__()

    def avgPriceAfter(self, date, routeId):
        sql = "SELECT routeId,departDate,price,date,routeLine from tb_route where date>='%s' and routeId='%d'" \
              % (date, routeId)
        self.cur.execute(sql)
        rets = self.cur.fetchall()
        num = 0
        sumPrice = 0
        for ret in rets:
            sumPrice += ret["price"]
            num += 1
        avgPrice = sumPrice / num
        return avgPrice

    def lowestPrice(self, date, routeId):
        sql = "SELECT min(price) from tb_route where date>='%s' and routeId='%d'" \
              % (date, routeId)
        self.cur.execute(sql)
        ret = self.cur.fetchone()

        return ret["min(price)"]

    # 把随机生成订单提前一次性写入数据库，游戏初始化时用
    def createOrder(self, routeId):
        sql_1 = "SELECT * from tb_route where routeId='%d'" % (routeId)
        self.cur.execute(sql_1)
        ret_1 = self.cur.fetchone()
        # 获取第一天日期、出发日期
        date = ret_1["date"]
        departDate = ret_1["departDate"]
        orderId = 1

        while date < departDate:
            ret_2 = self.nowRoute(date, routeId)
            todayPrice = ret_2.price

            ret_3 = self.allRoute(date, routeId)
            sumprice = 0
            num = 0
            for ret in ret_3:
                sumprice += ret["price"]
                num += 1
            avgprice = sumprice / num
            avgprice -= random.randint(0, 10)

            if avgprice < todayPrice:
                price = avgprice
            else:
                price = todayPrice - random.randint(0, 10)

            sql = "INSERT INTO tb_random_order(orderId, createDate, price) \
                    VALUES ('%s', '%s', '%s')" % \
                    (orderId, date, round(price))
            self.cur.execute(sql)
            self.conn.commit()
            orderId += 1
            date += timedelta(days=1)

    def lowerPriceAfter(self, date, routeId, price):
        sql = "SELECT * from tb_route where date>'%s' and routeId='%d' and price<'%d'" \
              % (date, routeId, price)
        self.cur.execute(sql)
        ret = self.cur.fetchall()

        return ret.__len__()

    # 将tb_random_order表清空，重置游戏时用
    def setEmpty(self):
        sql = "TRUNCATE TABLE tb_random_order"
        self.cur.execute(sql)
        self.conn.commit()

    # 从tb_random_order中查询生成的订单价格
    def getRandomOrder(self, date):
        sql = "SELECT price from tb_random_order where createDate ='%s'" % (date)
        newOrder = {"existence": 0, "price": 0}
        try:
            self.cur.execute(sql)
            ret = self.cur.fetchone()
            newOrder["existence"] = 1
            newOrder["price"] = ret["price"]
        except:
            newOrder["existence"] = 0

        return newOrder


# 对tb_order表进行读、写操作， 继承自ConnectMysql类
class OrderMysql(ConnectMysql):
    # 将新订单写入数据库
    def addOrder(self, order):
        # _datetime_departDate = datetime.strptime(order.getdepartDate(), '%Y-%m-%d')  # 变成日期格式
        # _datetime_createDate = datetime.strptime(order.getcreateDate(), '%Y-%m-%d')
        # datetime_departDate = _datetime_departDate.strftime('%Y-%m-%d')
        # datetime_createDate = _datetime_createDate.strftime('%Y-%m-%d')

        sql = "INSERT INTO tb_order(orderId, departDate, routeLine, price, createDate, isFinished, profit, maxProfit, avgProfit) \
               VALUES ('%s', '%s',  '%s',  '%s',  '%s', '%s', '%s', '%s','%s')" % \
               (order.getorderId(), order.getdepartDate(), order.getrouteLine(), order.getprice(),
                order.getcreateDate(), order.getisFinished(), order.getprofit(), order.maxProfit, order.avgProfit)
        self.cur.execute(sql)
        self.conn.commit()

    # 根据订单的orderId返回查找到底order类
    def findOrder(self, orderId):
        sql = "SELECT * FROM tb_order WHERE orderId = %s" % (orderId)

        try:
            # 执行SQL语句
            self.cur.execute(sql)
            # 获取数据
            result = self.cur.fetchone()
            # 数据变成一个order类并返回
            orderReturn = Order(result["orderId"], result["departDate"], result["routeLine"],
                                result["price"], result["createDate"], result["maxProfit"], result["avgProfit"])
            orderReturn.setisFinished(result["isFinished"])
            orderReturn.setprofit(result["profit"])

            return orderReturn
        except:
            print("Error: unable to fetch data")

    # 返回当前订单个数
    def findOrderNum(self):
        num = 0
        sql = "select * from tb_order"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        num = len(results)

        return num

    # 完成订单，修改订单状态、写入并返回收益、最大收益
    def finishOrder(self, orderId, priceBuy):
        priceAccept = self.findOrder(orderId).price
        # 收益
        profit = priceAccept - priceBuy

        sql = "UPDATE tb_order SET isFinished = 1, profit = '%s' WHERE orderId = '%s'" % \
              (profit, orderId)
        self.cur.execute(sql)
        self.conn.commit()

        return profit

    # 清除所有订单，重置游戏用
    def deleteAllOrder(self):
        sql = "TRUNCATE TABLE tb_order"
        self.cur.execute(sql)
        self.conn.commit()

    # 获取当前总收益
    def getProfitSum(self):
        sum = 0
        sql = "select isFinished, profit from tb_order"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        for result in results:
            if result["isFinished"] == 1:
                sum += result["profit"]
            # else:
               #  sum += -2000

        return sum

    # 获取当前平均收益
    def getAvgProfitSum(self):
        sum = 0
        sql = "select avgProfit from tb_order"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        for result in results:
            sum += result["avgProfit"]
        if sum == 0: # 防止报错
            sum = 1
        return sum

    # 获取当前最大收益
    def getMaxProfitSum(self):
        sum = 0
        sql = "select MaxProfit from tb_order '"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        for result in results:
            sum += result["avgProfit"]
        if sum == 0:  # 防止报错
            sum = 1
        return sum


    # 返回未完成订单价格列表
    def getOrderPriceRemain(self):
        priceDict = {}
        sql = "select * from tb_order"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        for result in results:
            if result["isFinished"] == 0:
                priceDict[result["orderId"]] = result["price"]

        return priceDict

#对tb_record进行写入
class RecordMysql(ConnectMysql):
    # 写入新record
    def newRecord(self, recordId, totalReward, priceBound,orderBound, frequency, punishment, routeId):
        sql = "INSERT INTO tb_record(recordId, totalReward, priceBound,orderBound, frequency, punishment, routeId)\
            VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s')"% \
              (recordId, totalReward, priceBound, orderBound, frequency, punishment, routeId)
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except:
            self.conn.rollback()

class GameRecordMysql(ConnectMysql):
    # 记录新订单
    def RecordAddOrder(self, order, gameNum, routeId):
        sql = "INSERT INTO game_record(gameNum, routeId, orderId, departDate, routeLine, price, createDate, isFinished, profit, maxProfit, avgProfit) \
               VALUES ('%s', '%s', '%s', '%s',  '%s',  '%s',  '%s', '%s', '%s', '%s', '%s')" % \
               (gameNum, routeId, order.getorderId(), order.getdepartDate(), order.getrouteLine(), order.getprice(),
                order.getcreateDate(), order.getisFinished(), order.getprofit(), order.maxProfit, order.avgProfit)
        self.cur.execute(sql)
        self.conn.commit()

    def getCreateDate(self, gameNum, orderId):
        sql = "SELECT createDate from game_record " \
              "where gameNum = '%d' and orderId = '%d' " \
              % (gameNum, orderId)
        try:
            self.cur.execute(sql)
            ret = self.cur.fetchone()
            createDate = ret["createDate"]
            return createDate
        except:
            print("Error: unable to fetch data")

    # 记录完成订单
    def RecordFinishOeder(self, gameNum, orderId, profit, finishDate):
        createDate = self.getCreateDate(gameNum, orderId)
        waitDateNum = (finishDate - createDate).days
        sql = "UPDATE game_record " \
              "SET isFinished = 1, profit = '%s', finishDate = '%s', waitDateNum = '%s' " \
              "WHERE gameNum = '%s' and orderId = '%s'" % \
              (profit, finishDate, waitDateNum, gameNum, orderId)
        self.cur.execute(sql)
        self.conn.commit()

    # 清除所有记录，重置游戏用
    def deleteAllRecord(self):
        sql = "TRUNCATE TABLE game_record"
        self.cur.execute(sql)
        self.conn.commit()

class MeanMysql(ConnectMysql):

    def Add(self, routeId, mean, std):
        sql = "INSERT INTO tb_mean_var(routeId, mean, std) \
                        VALUES ('%s', '%s', '%s')" % \
              (routeId, mean, std)
        self.cur.execute(sql)
        self.conn.commit()

    def Get(self, routeId):
        sql = "SELECT * From tb_mean_std WHERE routeId = '%s'" % (routeId)
        self.cur.execute(sql)
        self.conn.commit()
        ret = self.cur.fetchone()
        return ret["mean"], ret["std"]

if __name__ == '__main__':

    rm = RouteMysql()
    route_list = []
    one_route = []
    for i in range(1, 129, 1):
        if( rm.allRouteLength(i) < 87):
            print(str(i)  + " " + str(rm.allRouteLength(i)))
    for i in range(1, 129):  # 0 到 127 共 128 条
        one_route_list = rm.getOneRoute(i)
        route_list.append(one_route_list)

    for j in range(87):
        one_route.append(route_list[21][j].price)

    plt.plot(one_route)
    plt.savefig('./picture/' + "route22" + '_.png')
    plt.show()



    '''
    rm = RouteMysql()
    print(rm.lowerPriceAfter("2019-7-16", 2, 1600))
    '''

    gm = GameRecordMysql()
    gm.deleteAllRecord()

    '''
    mm = MeanMysql()
    rm = RouteMysql()
    a = np.empty(87 * 49)
    index = 0
    for i in range(1, 49, 1):
        results = rm.allRoute(i)
        for result in results:
            a[index] = result["price"]
            index += 1
    mean = np.mean(a)
    std = np.std(a)
    mm.Add(0, mean, std)
    '''
