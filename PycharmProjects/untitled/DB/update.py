#coding = "utf-8"
import MySQLdb

conn = MySQLdb.Connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test',
    charset='utf8'
)
cursor = conn.cursor()
sql_insert = "insert into user(username) values('haha')"
sql_update = "update user set username='hello1' where userid = 4"
sql_delete = "delete from user where userid<3"
try:
    cursor.execute(sql_insert)
    cursor.execute(sql_insert)
    print cursor.rowcount
    cursor.execute(sql_update)
    print cursor.rowcount
    cursor.execute(sql_delete)
    print cursor.rowcount
    conn.commit()  #事务提交
except Exception as e:
    print e
    conn.rollback() #事务回滚

cursor.close()
conn.close()