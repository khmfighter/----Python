import MySQLdb

conn = MySQLdb.Connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test',
    charset = 'utf8'
)
cursor = conn.cursor()
sql = "select * from user"
cursor.execute(sql)

print cursor.rowcount

#rs = cursor.fetchone()
#print rs

#rs = cursor.fetchmany(3)
#print rs

re = cursor.fetchall()
for re1 in re:
    print 'userid=%s,username=%s'%re1


cursor.close()
conn.close()