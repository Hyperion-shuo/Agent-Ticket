import pymysql
from datetime import datetime, date
from datetime import timedelta

class ConnectMysql():
    def __init__(self, host='localhost', port=3306, db='game', user='root', passwd='123456', charset='utf8'):
        # 建立连接
        self.conn = pymysql.connect(host=host, port=port, db=db, user=user, passwd=passwd, charset=charset)
        # 创建游标，操作设置为字典类型
        self.cur = self.conn.cursor(cursor = pymysql.cursors.DictCursor)

    #将航线数据从mysql写入到普通文件中，减少数据库交互的次数
    def routeExtract(self):
        content = []
        for i in range(1,50):
            sql = "SELECT * from tb_route where routeId='%s' ORDER BY date asc"%(i)
            self.cur.execute(sql)
            results = self.cur.fetchall()
            price = []
            start_date = ""
            end_date = ""
            for result in results:
                price.append(result["price"])
                if start_date == "":
                    start_date = result["date"].strftime("%Y-%m-%d")
                    end_date = result["departDate"].strftime("%Y-%m-%d")

            price.append(start_date)
            price.append(end_date)
            content.append(price)
        # print(content)
        self.fileWrite("F:\\MyWorkSpace\\docker28\\Ticket\\data\\route",content)
        self.conn.commit()

    # 将随机订单数据从mysql写入到普通文件中，减少数据库交互的次数
    def orderExtract(self):
        content = []
        sql = "SELECT * from tb_random_order ORDER BY createDate asc"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        price = 0
        create_date = ""
        for result in results:
            price = result["price"]
            create_date = result["createDate"].strftime("%Y-%m-%d")
            content.append([price,create_date])
        print(content)
        self.fileWrite("F:\\MyWorkSpace\\docker28\\Ticket\\data\\order", content)
        self.conn.commit()

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

    def fileWrite(self,filename,content):
        with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            for line in content:
                f.writelines(str(line)+"\n")
            print("write_end")

if __name__ == '__main__':
    rm = ConnectMysql()
    rm.orderExtract()