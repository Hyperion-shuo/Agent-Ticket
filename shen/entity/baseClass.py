
class Money:
    moneyId : int
    money : int
    today: str
    days: int


    def setMoneyId(self, moneyId):
        self.moneyId = moneyId


    def setMoney(self, money):
        self.money = money


    def setToday(self, today):
        self.today = today


    def setDays(self, days):
        self.days = days

    def getMoneyId(self):
        return self.moneyId


    def getMoney(self):
        return self.money


    def getToday(self):
        return self.today


    def getDays(self):
        return self.days


class Route:
    routeId: int
    departDate: str
    price: int
    date: str
    routeLine: str

    def setRouteId(self, routeId):
        self.routeId = routeId


    def setDepartDate(self, departDate):
        self.departDate = departDate


    def setPrice(self, price):
        self.price = price


    def setDate(self, date):
        self.date = date

    def setRouteLine(self, routeLine):
        self.routeLine = routeLine

    def getRouteId(self):
        return self.routeId


    def getDepartDate(self):
        return self.departDate


    def getPrice(self):
        return self.price


    def getDate(self):
        return self.date

    def getRouteLine(self):
        return self.routeLine


class Order:
    def __init__(self, orderId, departDate, routeLine, price, createDate, maxProfit, avgProfit):
        self.orderId = orderId
        self.departDate = departDate
        self.routeLine = routeLine
        self.price = price
        self.createDate = createDate
        self.isFinished = 0
        self.profit = 0
        self.maxProfit = maxProfit
        self.avgProfit = avgProfit

    def setcreateDate(self, createDate):
        self.createDate = createDate


    def setorderId(self, orderId):
        self.orderId = orderId


    def setdepartDate(self, departDate):
        self.departDate = departDate


    def setrouteLine(self, routeLine):
        self.routeLine = routeLine


    def setprice(self, price):
        self.price = price


    def setisFinished(self, isFinished):
        self.isFinished = isFinished


    def setprofit(self, profit):
        self.profit = profit


    def getcreateDate(self):
        return self.createDate


    def getorderId(self):
        return self.orderId


    def getdepartDate(self):
        return self.departDate


    def getrouteLine(self):
        return self.routeLine


    def getprice(self):
        return self.price


    def getisFinished(self):
        return self.isFinished


    def getprofit(self):
        return self.profit


class Record:
    recordId: int
    totalReward: int
    priceBound: int
    orderBound: int
    frequency: int
    punishment: int
    routeId: int

    def setRecorId(self , recordId):
        self.recordId = recordId

    def setTotalReward(self , totalReward):
        self.totalReward = totalReward

    def setPriceBound(self , priceBound):
        self.priceBound = priceBound

    def setOrderBound(self , orderBound):
        self.orderBound = orderBound

    def setFrequency(self, frequency):
        self.frequency = frequency

    def setPunishment(self, punishment):
        self.punishment = punishment

    def setRouteId(self, routeId):
        self.routeId = routeId

    def getRecorId(self):
        return self.routeId

    def getTotalReward(self):
        return self.totalReward

    def getPriceBound(self):
        return self.priceBound

    def getOrderBound(self):
        return self.orderBound

    def getFrequency(self):
        return self.frequency

    def getPunishment(self):
        return self.punishment

    def getRouteId(self):
        return self.routeId