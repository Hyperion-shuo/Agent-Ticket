import numpy as np

def OrderGenerator(data, mode):
    """
    :param data: 一个起飞日的87天价格list
    :param mode: 1，2 两个难度，2更难
    :return: 长为87的订单list 有订单的日期对应数值为订单价格， 没订单的日期对应-1

    订单的日期分布是没天1/5概率产生新订单
    """

    order_list = []
    order_price = -1
    if mode == 1:
        for i in range(87):
            have_order = np.random.randint(0, 3)
            if have_order == 0:
                order_price = np.min(np.hstack((np.average(data[:i]), data[i]))) - np.random.randint(0, 50)
            else:
                order_price = -1
            order_list.append(order_price)


    elif mode == 2:
        for i in range(87):
            have_order = np.random.randint(0, 3)
            if have_order == 0:
                order_price = np.min(np.hstack((np.average(data[:i]), data[i]))) * ((0.95 - 0.7) * np.random.sample() + 0.7)
            else:
                order_price = -1
            order_list.append(order_price)

    return order_list


def readRoute(filename):
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
                    list.append(c[i].replace(" ", ""))
            result.append(list)
        # print(result)
        return result

if __name__ == "__main__":
    route_list = readRoute("./wang/data/route")
    print(type(route_list))
    print(len(route_list[0]))
    print(route_list[1])