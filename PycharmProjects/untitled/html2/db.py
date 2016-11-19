import MySQLdb
conn = MySQLdb.Connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test',
    charset = 'utf8')


cursor = conn.cursor()

def addUser(username, password):
    sql = "insert into username(username,password) values('%s','%s')"%(username,password)
    cursor.execute(sql)
    conn.commit()
    conn.close()

def isExisted(username,password):
    sql = "select * from username where username='%s' and password ='%s'"%(username,password)
    cursor.execute(sql)
    result = cursor.fetchall()
    if(len(result) == 0):
        return False
    else:
        return True
