#coding=utf-8

from pymysql import *

"""
#打开数据库连接
db = connect(host="localhost" ,port=3306, user="root",passwd='123123', db='python3')
#使用cursor()方法创建一个游标对象
cursor = db.cursor()
#使用execute()方法执行SQL查询
sql = "insert into students values(0,'许会芬',0,'1995-08-11',1)"
sql1 = "delete from students where id=11"
paramList = ["曹旭",'1996-04-09']
sql2 = "insert into students(sname,birthday) values(%s,%s)"
cursor.execute(sql2,paramList)
db.commit() #因为默认是打开事务模式，所以要commit是执行生效
#使用fetchone()方法获取单条数据
#data = cursor.fetchall()
#print(data)
cursor.close()
db.close()
"""

#用类来封装
class MysqlOperation(object):
    def __init__(self,host,port,user,passwd,db,charset='utf8'): #有默认参数的要放在后面，否则报错
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset

    def ConnOpen(self):
        self.conn = connect(host=self.host,port=self.port,user=self.user,
                            passwd=self.passwd,db=self.db,charset=self.charset)
        self.cursor = self.conn.cursor()

    def ConnClose(self):
        self.cursor.close()
        self.conn.close()

    def CUD(self,sql,params):
        try:
            self.ConnOpen()

            self.cursor.execute(sql,params)
            self.conn.commit()

            self.ConnClose()

            print("OK")

        except Exception as e:
            print(e.message)

    def ReturnAll(self,sql,params):
        try:
            self.ConnOpen()

            self.cursor.execute(sql, params)
            result = self.cursor.fetchall()

            self.ConnClose()

            return result

        except Exception as e:
            print(e.message)

"""
params = ["飞飞","1998-09-24"]
sql = "insert into students(sname,birthday) values(%s,%s)"

mysql = MysqlOperation("localhost",3306,"root","123123","python3")
mysql.CUD(sql,params)

"""

"""
#小练习：用户登录
from hashlib import sha1 #python自带的加密包hashlib

#接受用户数输入
name = input("请输入用户名：")
pwd = input("请输入密码：")

#对密码加密
sh = sha1()
sh.update(b"pwd")
pwd2 = sh.hexdigest()
print("密码加密后的结果是：",pwd2)

#根据用户名查询密码
sql = "select passwd from users where name=%s"
mySql = MysqlOperation("localhost",3306,"root","123123","python3")
result = mySql.ReturnAll(sql,name) #查到就按元组返回(('37fa265330ad83eaa879efb1e2db6380896cf639'),)
#如果结果为空就返回空元组()
#print(result)

#也就是说如果返回的是空元组，则用户名不存在
if len(result)==0:
    print("用户名错误")

elif result[0][0]==pwd2:
    print("登录成功")

else:
    print("密码错误")

"""