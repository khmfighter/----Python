import MySQLdb

conn = MySQLdb.Connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test',
    charset = 'utf8')


cur = conn.cursor()

sql = 'select * from student'

cur.execute(sql)
result = cur.fetchall()


i=0
for row in result:
    print row


cur.close()
conn.close()